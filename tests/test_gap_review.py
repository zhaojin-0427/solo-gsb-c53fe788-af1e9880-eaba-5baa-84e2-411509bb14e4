"""镜头留白审阅回归测试。

场景 1（相接镜头 [0,1200)、[1200,2400)；cue1=100–500 镜头内，
cue2=900–1300 跨镜头，tol=200 下提案 800–1200）：
- 有效时间轴（采纳 → 提案 → 原值）下镜头 1 的留白为 [0,100)、[500,800)，
  镜头 2 无字幕覆盖、整段 [1200,2400) 一项；
- 标记「有意留白 / 待补字幕」后持久化，详情接口可见；
- 采纳 cue2（提案值）与同参数重算不改变留白边界，结论全部保留；
- 容差收紧到 50ms 后 cue2 无提案、有效时间轴回退原值 900–1300：
  边界未变的 [0,100) 保留结论，变化的区间回到待审（旧结论被剪除）；
- 过期页面提交旧边界被 409 拒绝，不会误标新留白。

场景 2（单镜头 [0,2400)，A=0–1000、B=1000–2000）：min_gap 40→30 后
提案漂移（B: 1040→1030）而采纳仍有效；撤销 B 的采纳使有效时间轴落到
新提案上，旧留白边界变化 → 结论剪除、新区间待审。

另覆盖：compute_shot_gaps 的裁剪/合并/整段/分别审阅语义、无镜头不生成
审阅项、撤销标记幂等，以及 JSON/SRT/ZIP 导出语义保持不变。
"""
import asyncio
import hashlib
import io
import json
import os
import sys
import zipfile

DB = "/tmp/test_gap_review.db"
if os.path.exists(DB):
    os.remove(DB)
os.environ["DATABASE_PATH"] = DB  # 须在导入 app 之前设置

sys.path.insert(0, "/workspace")

from fastapi import HTTPException, UploadFile  # noqa: E402

from app import database as db  # noqa: E402
from app import main as m  # noqa: E402
from app.gaps import compute_shot_gaps  # noqa: E402

db.init_db()

SRT = ("1\n00:00:00,100 --> 00:00:00,500\n镜头内字幕\n\n"
       "2\n00:00:00,900 --> 00:00:01,300\n跨镜头字幕\n")
SHOTS = json.dumps([{"start_ms": 0, "end_ms": 1200},
                    {"start_ms": 1200, "end_ms": 2400}])

SRT2 = ("1\n00:00:00,000 --> 00:00:01,000\n甲\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n乙\n")
SHOTS2 = json.dumps([{"start_ms": 0, "end_ms": 2400}])


def _upload(name: str, content: str) -> UploadFile:
    return UploadFile(io.BytesIO(content.encode("utf-8")), filename=name)


def _create(name, srt, shots=None, cps=21.0, gap=40, tol=200):
    return asyncio.run(m.create_project(
        name=name, cps_limit=cps, min_gap_ms=gap, tolerance_ms=tol,
        subtitle=_upload("a.srt", srt),
        shots_file=_upload("s.json", shots) if shots else None,
    ))


def _settings(pid, cps=21.0, gap=40, tol=200):
    return m.update_settings(pid, m.SettingsIn(
        cps_limit=cps, min_gap_ms=gap, tolerance_ms=tol))


def _mark(pid, shot_ord, start, end, decision):
    return m.mark_gap(pid, m.GapReviewIn(
        shot_ord=shot_ord, start_ms=start, end_ms=end, decision=decision))


def _gap_rows(proj):
    """[(shot_ord, start_ms, end_ms, decision)]，便于断言。"""
    return [(i["shot_ord"], i["start_ms"], i["end_ms"], i["decision"])
            for i in proj["gaps"]["items"]]


def _db_review_keys(pid):
    with db.get_conn() as conn:
        return {(r["shot_ord"], r["start_ms"], r["end_ms"])
                for r in db.get_gap_reviews(conn, pid)}


def _check_doc_hash(doc):
    got = doc["document_hash"]
    doc2 = dict(doc, document_hash="")
    raw = json.dumps(doc2, ensure_ascii=False, indent=2, sort_keys=True)
    assert hashlib.sha256(raw.encode("utf-8")).hexdigest() == got


def test_compute_shot_gaps_units():
    # 相接的覆盖段合并：只留尾部一段留白
    assert compute_shot_gaps([(1, 0, 1200)], [(0, 500), (500, 1000)]) == [
        {"shot_ord": 1, "shot_start_ms": 0, "shot_end_ms": 1200,
         "start_ms": 1000, "end_ms": 1200}]
    # 重叠覆盖合并并裁剪到镜头：整个镜头被盖满 → 无留白
    assert compute_shot_gaps([(1, 0, 1200)], [(-100, 600), (400, 1300)]) == []
    # 覆盖恰满镜头（端点相接）→ 无留白
    assert compute_shot_gaps([(1, 0, 1200)], [(0, 1200)]) == []
    # 完全没有字幕覆盖的镜头以整段为一项
    assert compute_shot_gaps([(1, 0, 1200)], []) == [
        {"shot_ord": 1, "shot_start_ms": 0, "shot_end_ms": 1200,
         "start_ms": 0, "end_ms": 1200}]
    # 零长/越界覆盖段被丢弃，同样整段一项
    assert compute_shot_gaps([(1, 0, 1200)], [(300, 300), (1500, 2000)]) == [
        {"shot_ord": 1, "shot_start_ms": 0, "shot_end_ms": 1200,
         "start_ms": 0, "end_ms": 1200}]
    # 无镜头 → 不生成审阅项
    assert compute_shot_gaps([], [(0, 100)]) == []
    # 相接镜头分别审阅：覆盖恰止于切点，镜头 1 无留白、镜头 2 整段留白
    items = compute_shot_gaps([(1, 0, 1200), (2, 1200, 2400)], [(0, 1200)])
    assert [(i["shot_ord"], i["start_ms"], i["end_ms"]) for i in items] == [
        (2, 1200, 2400)]
    # 重叠镜头分别审阅：同一覆盖各自裁剪，互不合并
    items = compute_shot_gaps([(1, 0, 1000), (2, 500, 1500)], [(0, 1000)])
    assert [(i["shot_ord"], i["start_ms"], i["end_ms"]) for i in items] == [
        (2, 1000, 1500)]
    # 覆盖段之间的空隙与镜头头尾空隙同时出现
    items = compute_shot_gaps([(1, 0, 1000)], [(100, 200), (400, 600)])
    assert [(i["start_ms"], i["end_ms"]) for i in items] == [
        (0, 100), (200, 400), (600, 1000)]
    print("compute_shot_gaps unit semantics ok")


def test_mark_persist_and_unchanged_boundaries_keep_decisions():
    proj = _create("留白主场景", SRT, SHOTS)
    pid = proj["id"]
    # 有效时间轴：cue1 提案 100–500，cue2 提案 800–1200
    assert _gap_rows(proj) == [
        (1, 0, 100, None),
        (1, 500, 800, None),
        (2, 1200, 2400, None),          # 无字幕镜头：整段一项
    ]
    assert proj["gaps"]["stats"] == {
        "total": 3, "pending": 3, "intentional": 0, "needs_subtitle": 0}
    # 每项都带镜头区间与时长，供页面列出起止时间
    it = proj["gaps"]["items"][1]
    assert (it["shot_start_ms"], it["shot_end_ms"]) == (0, 1200)
    assert it["duration_ms"] == 300

    proj = _mark(pid, 1, 0, 100, "intentional")
    assert proj["gaps"]["items"][0]["decision"] == "intentional"
    proj = _mark(pid, 1, 500, 800, "needs_subtitle")
    proj = _mark(pid, 2, 1200, 2400, "intentional")
    assert _gap_rows(proj) == [
        (1, 0, 100, "intentional"),
        (1, 500, 800, "needs_subtitle"),
        (2, 1200, 2400, "intentional"),
    ]
    assert proj["gaps"]["stats"] == {
        "total": 3, "pending": 0, "intentional": 2, "needs_subtitle": 1}

    # 详情接口（项目页读取）持久化可见
    detail = m.get_project(pid)
    assert _gap_rows(detail) == _gap_rows(proj)

    # 采纳 cue2（提案值 800–1200）：有效时间轴不变 → 结论全部保留
    cue2 = next(c for c in detail["analysis"]["cues"] if c["cue_id"] == "2")
    proj = m.adopt(pid, m.AdoptIn(cue_uid=cue2["uid"],
                                  new_start_ms=800, new_end_ms=1200))
    assert proj["analysis"]["cues"][1]["adopted"] is True
    assert _gap_rows(proj) == [
        (1, 0, 100, "intentional"),
        (1, 500, 800, "needs_subtitle"),
        (2, 1200, 2400, "intentional"),
    ]

    # 相同参数重算：边界不变 → 结论保留
    proj = _settings(pid, gap=40, tol=200)
    assert _gap_rows(proj) == [
        (1, 0, 100, "intentional"),
        (1, 500, 800, "needs_subtitle"),
        (2, 1200, 2400, "intentional"),
    ]
    print("mark/persist + unchanged boundaries keep decisions ok")
    return pid


def test_settings_change_migrates_and_stale_rejected(pid):
    # 容差收紧到 50ms：cue2 无提案，有效时间轴回退原值 900–1300
    proj = _settings(pid, gap=40, tol=50)
    assert proj["analysis"]["cues"][1]["proposal_start_ms"] is None
    # [0,100) 边界未变 → 结论保留；[500,800)→[500,900)、
    # [1200,2400)→[1300,2400) 边界变化 → 回到待审
    assert _gap_rows(proj) == [
        (1, 0, 100, "intentional"),
        (1, 500, 900, None),
        (2, 1300, 2400, None),
    ]
    assert proj["gaps"]["stats"] == {
        "total": 3, "pending": 2, "intentional": 1, "needs_subtitle": 0}
    # 库中只剩边界未变的那一条结论（迁移在同一事务内完成）
    assert _db_review_keys(pid) == {(1, 0, 100)}

    # 过期页面提交旧边界 [500,800)：必须 409 拒绝，且不得误标新区间
    try:
        _mark(pid, 1, 500, 800, "intentional")
        raise AssertionError("过期提交应被 409 拒绝")
    except HTTPException as exc:
        assert exc.status_code == 409, exc.status_code
    proj = m.get_project(pid)
    assert _gap_rows(proj) == [
        (1, 0, 100, "intentional"),
        (1, 500, 900, None),            # 未被误标
        (2, 1300, 2400, None),
    ]
    assert _db_review_keys(pid) == {(1, 0, 100)}

    # 从未存在过的区间同样 409；非法 decision 400
    try:
        _mark(pid, 1, 10, 20, "intentional")
        raise AssertionError("不存在的留白应被 409 拒绝")
    except HTTPException as exc:
        assert exc.status_code == 409
    try:
        _mark(pid, 1, 0, 100, "whatever")
        raise AssertionError("非法 decision 应被 400 拒绝")
    except HTTPException as exc:
        assert exc.status_code == 400

    # 新边界可以正常标记；撤销标记回到待审，且撤销幂等
    proj = _mark(pid, 1, 500, 900, "needs_subtitle")
    assert proj["gaps"]["items"][1]["decision"] == "needs_subtitle"
    proj = m.unmark_gap(pid, 1, 500, 900)
    assert proj["gaps"]["items"][1]["decision"] is None
    proj = m.unmark_gap(pid, 1, 500, 900)   # 幂等
    assert proj["gaps"]["items"][1]["decision"] is None
    print("settings migration + stale rejection + unmark ok")


def test_revert_changes_boundaries_migrates():
    """min_gap 收紧后提案漂移而采纳仍有效；撤销采纳改变有效时间轴，
    旧留白结论被剪除、新区间回到待审。"""
    proj = _create("撤销迁移场景", SRT2, SHOTS2, gap=40, tol=200)
    pid = proj["id"]
    assert _gap_rows(proj) == [(1, 1000, 1040, None), (1, 2040, 2400, None)]
    cues = proj["analysis"]["cues"]
    # 采纳两条提案（A: 0–1000，B: 1040–2040）
    proj = m.adopt(pid, m.AdoptIn(cue_uid=cues[0]["uid"],
                                  new_start_ms=0, new_end_ms=1000))
    proj = m.adopt(pid, m.AdoptIn(cue_uid=cues[1]["uid"],
                                  new_start_ms=1040, new_end_ms=2040))
    assert _gap_rows(proj) == [(1, 1000, 1040, None), (1, 2040, 2400, None)]

    # min_gap 40→30：B 的提案漂移为 1030–2030，两条采纳仍有效（40≥30）
    proj = _settings(pid, gap=30, tol=200)
    acues = proj["analysis"]["cues"]
    assert acues[0]["adopted"] and acues[1]["adopted"]
    assert acues[1]["proposal_start_ms"] == 1030
    # 有效时间轴仍取采纳值 → 留白边界不变
    assert _gap_rows(proj) == [(1, 1000, 1040, None), (1, 2040, 2400, None)]

    proj = _mark(pid, 1, 1000, 1040, "needs_subtitle")
    proj = _mark(pid, 1, 2040, 2400, "intentional")

    # 撤销 B 的采纳：有效时间轴落到新提案 1030–2030，两段留白边界都变
    proj = m.revert_adoption(pid, acues[1]["uid"])
    assert _gap_rows(proj) == [(1, 1000, 1030, None), (1, 2030, 2400, None)]
    assert _db_review_keys(pid) == set()
    # 过期页面再提交旧边界 → 409
    try:
        _mark(pid, 1, 2040, 2400, "intentional")
        raise AssertionError("过期提交应被 409 拒绝")
    except HTTPException as exc:
        assert exc.status_code == 409
    print("revert migration ok")


def test_no_shots_no_review_items():
    proj = _create("无镜头场景", SRT2, shots=None)
    pid = proj["id"]
    assert proj["shots"] == []
    assert proj["gaps"]["items"] == []
    assert proj["gaps"]["stats"] == {
        "total": 0, "pending": 0, "intentional": 0, "needs_subtitle": 0}
    # 无镜头时任何留白标记都不存在 → 409
    try:
        _mark(pid, 1, 0, 100, "intentional")
        raise AssertionError("无镜头项目不应有可标记留白")
    except HTTPException as exc:
        assert exc.status_code == 409
    print("no shots → no review items ok")


def test_exports_semantics_unchanged():
    proj = _create("导出场景", SRT, SHOTS)
    pid = proj["id"]
    _mark(pid, 1, 0, 100, "intentional")

    # JSON 导出：不含 gaps 字段，document_hash 仍按原结构校验
    doc = json.loads(m.export(pid, fmt="json").body.decode("utf-8"))
    assert "gaps" not in doc
    assert set(doc.keys()) == {
        "schema", "project", "document_hash", "cues", "conflicts", "stats"}
    _check_doc_hash(doc)
    assert doc["cues"][0]["status"] == "proposed"
    assert doc["cues"][1]["proposal"] == {"start_ms": 800, "end_ms": 1200}

    # 独立 SRT 与 ZIP 内 SRT 一致，时间轴不受留白审阅影响
    srt_text = m.export(pid, fmt="srt").body.decode("utf-8")
    assert "00:00:00,100 --> 00:00:00,500" in srt_text
    assert "00:00:00,800 --> 00:00:01,200" in srt_text
    zbody = m.export(pid, fmt="zip").body
    with zipfile.ZipFile(io.BytesIO(zbody)) as zf:
        assert zf.read("corrected.srt").decode("utf-8") == srt_text
        zdoc = json.loads(zf.read("correction.json").decode("utf-8"))
    assert "gaps" not in zdoc
    _check_doc_hash(zdoc)
    print("exports unchanged ok")


def main():
    test_compute_shot_gaps_units()
    pid = test_mark_persist_and_unchanged_boundaries_keep_decisions()
    test_settings_change_migrates_and_stale_rejected(pid)
    test_revert_changes_boundaries_migrates()
    test_no_shots_no_review_items()
    test_exports_semantics_unchanged()
    print("gap review regression ok")


if __name__ == "__main__":
    main()
