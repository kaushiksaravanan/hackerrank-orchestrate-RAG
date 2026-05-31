"""Architecture-diagram collision verifier.

Run after any change to docs/architecture.* (or canvas state) to confirm
no shape/text/arrow visually overlaps an unrelated element. Catches the
class of bug where a long arrow grazes a dotted callout box.

Checks:
  1. Shape-shape overlap (rectangles, ellipses, diamonds vs each other)
  2. Free-floating text overlap with any unrelated shape
  3. Arrow segments crossing unrelated boxes (Liang-Barsky line clipping
     with a 4-px inward margin so an arrow that just touches the boundary
     is not flagged; only true crossings fail)

Usage:
  # against the live canvas server
  curl -s http://localhost:3000/api/elements > /tmp/canvas.json
  python docs/check_collisions.py /tmp/canvas.json

  # against an exported scene
  python docs/check_collisions.py path/to/scene.json

Exits 0 on clean, 1 if any issues are found (suitable for CI).
"""
import json, sys

def load(path):
    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, list): return d
    return d.get("elements", [])

def bbox(e):
    return (e["x"], e["y"], e["x"]+e.get("width",0), e["y"]+e.get("height",0))

def overlaps(a, b, margin=0):
    ax1, ay1, ax2, ay2 = bbox(a)
    bx1, by1, bx2, by2 = bbox(b)
    return not (ax2<=bx1-margin or bx2<=ax1-margin or ay2<=by1-margin or by2<=ay1-margin)

def edge_point(shape, tx, ty):
    x, y = shape["x"], shape["y"]
    w, h = shape.get("width",0) or 0, shape.get("height",0) or 0
    cx, cy = x + w/2, y + h/2
    dx, dy = tx-cx, ty-cy
    if dx == 0 and dy == 0: return cx, cy
    t = shape.get("type")
    if t == "rectangle":
        rx = (w/2)/abs(dx) if dx else 1e9
        ry = (h/2)/abs(dy) if dy else 1e9
        r = min(rx, ry)
        return cx + dx*r, cy + dy*r
    if t == "ellipse":
        a, b = w/2, h/2
        denom = (dx/a)**2 + (dy/b)**2
        if denom <= 0: return cx, cy
        return cx + dx/denom**0.5, cy + dy/denom**0.5
    if t == "diamond":
        a, b = w/2, h/2
        denom = abs(dx)/a + abs(dy)/b
        if denom <= 0: return cx, cy
        return cx + dx/denom, cy + dy/denom
    return cx, cy

def arrow_segments(arrow, by_id):
    s = arrow.get("startBinding") or arrow.get("start") or {}
    e = arrow.get("endBinding") or arrow.get("end") or {}
    s_id = s.get("elementId") or s.get("id")
    e_id = e.get("elementId") or e.get("id")
    s_shape = by_id.get(s_id)
    e_shape = by_id.get(e_id)
    pts = arrow.get("points", [[0,0]])
    ax, ay = arrow.get("x",0), arrow.get("y",0)
    abs_pts = [(ax+p[0], ay+p[1]) for p in pts]
    if s_shape and len(abs_pts) >= 2:
        ex, ey = abs_pts[1]
        abs_pts[0] = edge_point(s_shape, ex, ey)
    if e_shape and len(abs_pts) >= 2:
        sx, sy = abs_pts[-2]
        abs_pts[-1] = edge_point(e_shape, sx, sy)
    return list(zip(abs_pts[:-1], abs_pts[1:])), s_id, e_id

def seg_rect_intersect(p1, p2, rect, margin=4):
    """Liang-Barsky line clipping. Margin shrinks the rect inward so that
    arrows touching the boundary do not register as crossings."""
    x1, y1 = p1; x2, y2 = p2
    rx1 = rect["x"] + margin
    ry1 = rect["y"] + margin
    rx2 = rect["x"] + rect.get("width",0) - margin
    ry2 = rect["y"] + rect.get("height",0) - margin
    if rx2 <= rx1 or ry2 <= ry1: return False
    dx, dy = x2-x1, y2-y1
    p = [-dx, dx, -dy, dy]
    q = [x1-rx1, rx2-x1, y1-ry1, ry2-y1]
    u1, u2 = 0.0, 1.0
    for i in range(4):
        if p[i] == 0:
            if q[i] < 0: return False
        else:
            t = q[i] / p[i]
            if p[i] < 0: u1 = max(u1, t)
            else: u2 = min(u2, t)
    return u1 < u2 - 1e-6

def main(path):
    els = load(path)
    by_id = {e["id"]: e for e in els}
    text_label = {t.get("containerId"): t.get("text","").replace("\n"," | ")[:40]
                  for t in els if t.get("type")=="text" and t.get("containerId")}
    label = lambda e: text_label.get(e["id"], e["id"][:14])

    boxes = [e for e in els if e.get("type") in ("rectangle","ellipse","diamond")]
    arrows = [e for e in els if e.get("type") == "arrow"]
    free_texts = [e for e in els if e.get("type") == "text" and not e.get("containerId")]

    issues = []

    for i, a in enumerate(boxes):
        for b in boxes[i+1:]:
            if overlaps(a, b):
                issues.append(("BOX-BOX", f"{label(a)} <> {label(b)}"))

    for t in free_texts:
        for b in boxes:
            if overlaps(t, b, margin=-2):
                issues.append(("TEXT-BOX", f"text '{t.get('text','')[:30]}' inside {label(b)}"))

    for arr in arrows:
        segs, s_id, e_id = arrow_segments(arr, by_id)
        for box in boxes:
            if box["id"] in (s_id, e_id): continue
            for p1, p2 in segs:
                if seg_rect_intersect(p1, p2, box):
                    src = label(by_id.get(s_id, {"id": s_id or "?"}))[:25]
                    dst = label(by_id.get(e_id, {"id": e_id or "?"}))[:25]
                    issues.append(("ARROW-BOX",
                        f"[{src} -> {dst}] crosses [{label(box)}]"))
                    break

    if not issues:
        print("OK: no collisions")
        return 0

    print(f"FAIL: {len(issues)} issue(s)")
    for kind, msg in issues:
        print(f"  {kind}: {msg}")
    return 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python check_collisions.py <canvas.json>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
