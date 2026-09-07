"""Reading-order sorting for detected OCR boxes.
Persian/Arabic reading order:
  1. Text lines are ordered top -> bottom.
  2. Within a line, boxes are ordered right -> left.

"Same line" is decided by vertical interval containment:
    top_B >= top_A and bottom_B <= bottom_A
    (or the symmetric case: A contained inside B)
"""

from typing import Any, Dict, List, Sequence, Tuple


def _bounds(polygon) -> Tuple[float, float, float, float]:
    xs = [float(pt[0]) for pt in polygon]
    ys = [float(pt[1]) for pt in polygon]
    return min(ys), max(ys), min(xs), max(xs)


def _is_same_line(top_a: float, bottom_a: float, top_b: float, bottom_b: float) -> bool:
    a_contains_b = top_b >= top_a and bottom_b <= bottom_a
    b_contains_a = top_a >= top_b and bottom_a <= bottom_b
    return a_contains_b or b_contains_a


def sort_boxes_reading_order(boxes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    n = len(boxes)
    if n <= 1:
        return list(boxes)

    bounds = [_bounds(b["polygon"]) for b in boxes]

    # Union-Find grouping for lines
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        top_i, bottom_i, _, _ = bounds[i]
        for j in range(i + 1, n):
            top_j, bottom_j, _, _ = bounds[j]
            if _is_same_line(top_i, bottom_i, top_j, bottom_j):
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    # Sort lines top -> bottom, boxes within a line right -> left
    lines: List[Tuple[float, List[int]]] = []
    for indices in groups.values():
        line_top = min(bounds[i][0] for i in indices)
        indices_sorted = sorted(indices, key=lambda i: bounds[i][3], reverse=True)
        lines.append((line_top, indices_sorted))

    lines.sort(key=lambda item: item[0])

    ordered: List[Dict[str, Any]] = []
    for _, indices_sorted in lines:
        for i in indices_sorted:
            ordered.append(boxes[i])

    return ordered
