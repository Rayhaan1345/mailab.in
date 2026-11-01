"""Offline script for traces.json: runs four short programs with sys.settrace and records each line executed. Run with python -I."""
import sys, json, io, contextlib, re
import os
R = os.path.dirname(os.path.abspath(__file__))
PROGS = [
 ("p1", "Add up the long sharks", """total = 0
for length in [42.4, 74.5, 64.3]:
    if length > 60:
        total = total + length
print(total)""", "total is 74.5 + 64.3 = 138.8. The loop header changes length each time and the if test decides whether line 4 runs."),
 ("p2", "Find the longest", """lengths = [42.4, 45.1, 74.5, 41.0]
best = lengths[0]
i = 0
while i < len(lengths):
    if lengths[i] > best:
        best = lengths[i]
    i = i + 1
print(best)""", "best only changes when a longer length turns up, here at i = 1 and i = 2. The line i = i + 1 runs on every pass, which is why the loop ends."),
 ("p3", "Binary search", """items = [28.5, 36.1, 41.0, 44.0, 54.4, 62.2, 75.0]
target = 54.4
lo = 0
hi = 6
steps = 0
found = -1
while lo <= hi:
    mid = (lo + hi) // 2
    steps = steps + 1
    if items[mid] == target:
        found = mid
        break
    elif items[mid] < target:
        lo = mid + 1
    else:
        hi = mid - 1
print(found, steps)""", "The window shrinks from positions 0 to 6, to 4 to 6, to 4 to 4. Three steps find 54.4 at position 4, as the formula floor(log2 7) + 1 = 3 allows."),
 ("p4", "Find the bug", """lengths = [74.5, 41.0, 36.1, 64.3]
count = 0
for i in range(1, len(lengths)):
    if lengths[i] > 40:
        count = count + 1
print(count)""", "The program should count the lengths over 40, which are 74.5, 41.0 and 64.3, so the answer should be 3. The loop starts at i = 1, so lengths[0] is never tested. Starting from range(len(lengths)) fixes it."),
]
def literal_rhs(src):
    m = re.match(r"^\s*\w+ = (.*)$", src)
    return bool(m) and bool(re.match(r"^(\[[^\]]*\]|-?[\d.]+)$", m.group(1).strip()))
def snap(d):
    out = {}
    for k, v in d.items():
        if k.startswith("__"): continue
        if isinstance(v, (int, float, bool, str)): out[k] = v
        elif isinstance(v, list): out[k] = v
    return out
progs = []
for pid, title, code, note in PROGS:
    lines = code.split("\n"); events = []
    g = {"__builtins__": __builtins__}
    buf = io.StringIO()
    def tracer(frame, event, arg):
        if frame.f_code.co_filename != "<prog>": return tracer
        if event == "line":
            events.append((frame.f_lineno, snap(frame.f_locals), buf.getvalue()))
        return tracer
    sys.settrace(tracer)
    with contextlib.redirect_stdout(buf):
        exec(compile(code, "<prog>", "exec"), g)
    sys.settrace(None)
    final = (None, snap(g), buf.getvalue())
    events.append(final)
    steps = []
    for k in range(len(events) - 1):
        line, before, out0 = events[k]; nline, after, out1 = events[k + 1]
        changes = {n: (v if isinstance(v, list) else repr(v)) for n, v in after.items() if n not in before or before[n] != v}
        src = lines[line - 1]
        new_out = out1[len(out0):].rstrip("\n") if out1 != out0 else None
        kind = "print" if new_out is not None else ("assign" if changes else "branch")
        steps.append({"line": line, "kind": kind, "changes": changes, "next": nline, "out": new_out, "auto": bool(changes) and literal_rhs(src)})
    scalars = []
    for s in steps:
        for n, v in s["changes"].items():
            if not isinstance(v, list) and n not in scalars: scalars.append(n)
    progs.append({"id": pid, "title": title, "lines": lines, "steps": steps, "vars": scalars, "note": note, "output": buf.getvalue().strip()})
    print(pid, len(steps), "steps", progs[-1]["output"], scalars, sum(s["auto"] for s in steps), "auto")
json.dump({"programs": progs}, open(f"{R}/traces.json", "w"), indent=0)
