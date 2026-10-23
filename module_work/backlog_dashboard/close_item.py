"""Move one backlog item to Done (run from the repo root): python module_work/backlog_dashboard/close_item.py ID "done text".

BACKLOG.md keeps CRLF line endings in the repo; the new line follows the
file's own ending. Newest entry first, straight after the section's intro
line (the section is empty right after a sweep)."""
import io
import sys

item_id, done_text = sys.argv[1], sys.argv[2]
p = "BACKLOG.md"
s = io.open(p, encoding="utf-8", newline="").read()
start = s.index(f"- **{item_id}** [")
end = s.index("\n", start) + 1
s = s[:start] + s[end:]
h = s.index("## Done since the last sweep")
intro = s.index("Older docs still list these as open.", h)
eol = "\r\n" if s[h:intro].count("\r\n") else "\n"
pos = s.index("\n", intro) + 1            # end of the intro line
if s.startswith(eol, pos):                 # its blank line
    pos += len(eol)
else:
    s = s[:pos] + eol + s[pos:]
    pos += len(eol)
s = s[:pos] + f"- **{item_id}** {done_text}{eol}" + s[pos:]
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("closed", item_id)
