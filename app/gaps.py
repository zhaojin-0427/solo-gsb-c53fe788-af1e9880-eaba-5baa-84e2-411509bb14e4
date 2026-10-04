"""镜头留白计算。

按导出 SRT 共用的有效时间轴（已采纳取采纳值，未采纳先取提案、无提案
取原值），把每条字幕的覆盖裁剪到各镜头内，合并重叠或相接的覆盖段，
求每个镜头未被字幕覆盖的半开毫秒区间 [start_ms, end_ms)。

相接或重叠的镜头各自独立计算、分别审阅（不做镜头合并）；完全没有
字幕覆盖的镜头以整段为一项；无镜头时不产生任何审阅项。

审阅项身份 = (shot_ord, start_ms, end_ms)：采纳、撤销或参数重算改变
有效时间轴后，边界未变的留白保留原结论，边界变化的旧结论被剪除，
对应新区间回到「待审」。
"""
from __future__ import annotations


def compute_shot_gaps(
    shots: list[tuple[int, int, int]],
    coverage: list[tuple[int, int]],
) -> list[dict]:
    """计算每个镜头内未被字幕覆盖的半开毫秒区间。

    shots: [(shot_ord, start_ms, end_ms)]，相接/重叠镜头分别处理；
    coverage: 有效时间轴上的字幕区间 [(start_ms, end_ms)]，可重叠、可乱序。
    返回按 (镜头起点, shot_ord, 留白起点) 排列的 dict 列表，每个元素含
    shot_ord / shot_start_ms / shot_end_ms / start_ms / end_ms。
    """
    items: list[dict] = []
    for shot_ord, s0, e0 in sorted(shots, key=lambda s: (s[1], s[0])):
        # 字幕覆盖裁剪到本镜头（丢弃空段）
        clipped: list[tuple[int, int]] = []
        for s, e in coverage:
            cs, ce = max(s, s0), min(e, e0)
            if cs < ce:
                clipped.append((cs, ce))
        clipped.sort()
        # 合并重叠或相接（s <= 已合并段的右端）的覆盖段
        merged: list[list[int]] = []
        for s, e in clipped:
            if merged and s <= merged[-1][1]:
                if e > merged[-1][1]:
                    merged[-1][1] = e
            else:
                merged.append([s, e])
        # 覆盖段的补集即留白（半开区间）；无覆盖时整段一项
        cursor = s0
        for s, e in merged:
            if s > cursor:
                items.append(_gap(shot_ord, s0, e0, cursor, s))
            if e > cursor:
                cursor = e
        if cursor < e0:
            items.append(_gap(shot_ord, s0, e0, cursor, e0))
    return items


def _gap(shot_ord: int, s0: int, e0: int, start_ms: int, end_ms: int) -> dict:
    return {
        "shot_ord": shot_ord,
        "shot_start_ms": s0,
        "shot_end_ms": e0,
        "start_ms": start_ms,
        "end_ms": end_ms,
    }
