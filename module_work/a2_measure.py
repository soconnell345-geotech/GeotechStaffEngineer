"""A2 context-economics measurement.

Models per-turn INPUT tokens for a 10-turn calc-heavy geotech session under
three context strategies, using the REAL Claude tokenizer (anthropic
count_tokens) on realistic content blocks, with a char/3.7 fallback.

Strategies:
  (a) full persist+replay  -- today: the whole message history is re-sent each turn.
  (b) summarization ON     -- deepagents SummarizationMiddleware; compacts older
                              history to a summary + keep-last-8 once input crosses
                              the trigger.  Shown at the DEFAULT trigger
                              (0.8 * 200k window) and an AGGRESSIVE trigger (30k).
  (c) calc-subagent        -- tool-heavy calc work runs in a subagent whose trace
                              (big tool results) never enters the main thread;
                              main thread sees only a compact return per calc.
"""
import os

MODEL_WINDOW = 200_000            # Opus 4.8 / Sonnet 5 input window
SYSTEM_OVERHEAD = 6000            # system prompt + tool schemas (fixed each turn)

# ---- tokenizer: real Claude counts if a key is present, else char/3.7 ----
_client = None
def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    return _client

_cache = {}
def ntok(text: str) -> int:
    if text in _cache:
        return _cache[text]
    n = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            r = _get_client().messages.count_tokens(
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": text}])
            n = int(r.input_tokens) - 8      # subtract ~fixed wrapper overhead
        except Exception as e:
            print("  [count_tokens failed, using char proxy]", str(e)[:80])
    if n is None:
        n = round(len(text) / 3.7)
    n = max(n, 1)
    _cache[text] = n
    return n

# ---- realistic content blocks (representative geotech calc-heavy turn) ----
USER_Q = ("Now check the same footing for the drained long-term case with the "
          "water table at 1.5 m, and compare Vesic vs Meyerhof. If it fails, "
          "tell me the required width.")

ASSIST_REASON = ("I'll run bearing capacity for the drained case with the "
                 "adjusted groundwater depth using both methods, then compare "
                 "against the demand and, if needed, solve for the width that "
                 "meets FS>=3. Let me call the bearing_capacity agent twice and "
                 "tabulate.") * 2

TOOL_CALL = ('{"agent":"bearing_capacity","method":"analyze_bearing_capacity",'
             '"params":{"B":2.0,"L":2.0,"D":1.0,"gamma":18.5,"phi":32,"c":5,'
             '"gwt_depth":1.5,"method":"vesic","surcharge":0}}')

# a calc-package / full method dump result -- the bulk payload of a calc turn
_CALC_ROW = ("| Nc=35.49 Nq=23.18 Ngamma=30.22 | sc=1.31 sq=1.25 sgamma=0.60 | "
             "dc=1.15 dq=1.10 dgamma=1.00 | ic=iq=igamma=1.00 | qult=1284.6 kPa "
             "| qall=428.2 kPa | FS=3.00 | notes: GWT reduces gamma' below base |\n")
CALC_RESULT = ("BEARING CAPACITY CALC PACKAGE (Vesic 1973) -- drained\n"
               "Inputs echoed; factors, depth/shape/incl corrections, GWT case:\n"
               + _CALC_ROW * 55 +
               "Basis: FHWA/NAVFAC DM7.2; effective stress; Terzaghi-Vesic "
               "superposition. Intermediate values retained for audit.\n")

METHOD_DUMP = ("describe_method(bearing_capacity.analyze_bearing_capacity):\n"
               "params: B,L,D,gamma,phi,c,gwt_depth,method,surcharge,"
               "load_incl,base_incl,...\n"
               "allowed method values: vesic|meyerhof|hansen|cbear ...\n"
               + ("  - detailed parameter description line with units and "
                  "defaults and cross-refs to DM7 tables.\n") * 40)

REF_EXCERPT = ("Reference excerpt (NAVFAC DM7.2, Ch.4, bearing capacity): "
               "for cohesionless soils the ultimate capacity ... "
               ) + ("effective-stress correction paragraph with equations and "
                    "table lookups. ") * 60

ASSIST_FINAL = ("Drained long-term (GWT at 1.5 m): Vesic q_all = 428 kPa, "
                "Meyerhof q_all = 401 kPa; both exceed the 350 kPa demand, "
                "FS 3.0+. The 2.0 m footing is adequate. If demand rose to "
                "500 kPa you'd need B ~= 2.4 m (Vesic). Table and calc package "
                "saved. ") * 6

# compact return the calc-SUBAGENT would hand back to the main thread
SUBAGENT_RETURN = ("[calc subagent] Vesic q_all=428 kPa, Meyerhof=401 kPa, "
                   "FS>=3.0 at B=2.0 m; required B=2.4 m if demand=500 kPa. "
                   "Calc package: files/bearing_drained.html.") * 2

# summary block the SummarizationMiddleware writes when it compacts
SUMMARY_BLOCK = ("[conversation summary] Site: 2 m sq footing, medium-dense "
                 "sand phi=32, c=5, gamma=18.5, GWT 1.5 m. Prior turns: "
                 "undrained & drained bearing checks (Vesic/Meyerhof), "
                 "settlement estimate, and a footing-width solve. Key accepted "
                 "values retained. ") * 4

# ---- per-message token sizes ----
t_user   = ntok(USER_Q)
t_reason = ntok(ASSIST_REASON)
t_tcall  = ntok(TOOL_CALL)
t_calc   = ntok(CALC_RESULT)
t_method = ntok(METHOD_DUMP)
t_ref    = ntok(REF_EXCERPT)
t_final  = ntok(ASSIST_FINAL)
t_subret = ntok(SUBAGENT_RETURN)
t_summ   = ntok(SUMMARY_BLOCK)

# A calc-heavy turn's NEW history when tools run IN the main thread (a & b):
#   user + reasoning + 2 tool calls + calc result + method dump + ref excerpt + final
per_turn_main = (t_user + t_reason + 2*t_tcall + t_calc + t_method + t_ref + t_final)
# When calc runs in a SUBAGENT (c): main thread only gains user + a compact
# subagent return + final answer (big tool results stay in the subagent).
per_turn_sub  = (t_user + t_subret + t_final)

print("=== per-message token sizes (real Claude tokenizer) ===")
for k,v in [("user_q",t_user),("assistant_reasoning",t_reason),("tool_call",t_tcall),
            ("calc_result",t_calc),("method_dump",t_method),("ref_excerpt",t_ref),
            ("assistant_final",t_final),("subagent_return",t_subret),
            ("summary_block",t_summ)]:
    print(f"  {k:20s} {v:6d}")
print(f"  NEW history per calc-turn (tools in main thread): {per_turn_main}")
print(f"  NEW history per calc-turn (calc in subagent):     {per_turn_sub}")
print()

NTURNS = 10

def replay_curve(per_turn_new):
    """Full persist+replay: input(t) = system + sum(new history of turns < t) + this turn's new."""
    inputs = []
    hist = 0
    for t in range(1, NTURNS+1):
        inp = SYSTEM_OVERHEAD + hist + per_turn_new
        inputs.append(inp)
        hist += per_turn_new
    return inputs

def summarize_curve(per_turn_new, trigger, keep_msgs_tokens):
    """Summarization: replay until input crosses trigger, then compact prior
    history to SUMMARY + keep-last (~keep_msgs_tokens), and continue."""
    inputs = []
    hist = 0
    summarized_base = 0   # tokens of summary standing in for evicted history
    for t in range(1, NTURNS+1):
        raw_in = SYSTEM_OVERHEAD + summarized_base + hist + per_turn_new
        if raw_in > trigger:
            # compact: evicted history -> one summary block; keep recent tail
            summarized_base = t_summ
            hist = keep_msgs_tokens
            raw_in = SYSTEM_OVERHEAD + summarized_base + hist + per_turn_new
        inputs.append(raw_in)
        hist += per_turn_new
    return inputs

a  = replay_curve(per_turn_main)
b_default = summarize_curve(per_turn_main, trigger=int(0.8*MODEL_WINDOW), keep_msgs_tokens=8*per_turn_main//4)
b_aggr    = summarize_curve(per_turn_main, trigger=30000, keep_msgs_tokens=2*per_turn_main)
c  = replay_curve(per_turn_sub)          # subagent + today's replay
c_ckpt = [SYSTEM_OVERHEAD + per_turn_sub]*NTURNS  # subagent + durable state (no replay): flat-ish

print("=== per-turn INPUT tokens ===")
print(f"{'turn':>4} | {'(a) replay':>11} | {'(b) sum@160k':>12} | {'(b) sum@30k':>11} | {'(c) subagent':>12} | {'(c)+ckpt':>9}")
for i in range(NTURNS):
    print(f"{i+1:>4} | {a[i]:>11,} | {b_default[i]:>12,} | {b_aggr[i]:>11,} | {c[i]:>12,} | {c_ckpt[i]:>9,}")

print()
print("=== cumulative INPUT tokens over the 10-turn session (what you pay for) ===")
print(f"  (a) full replay            : {sum(a):>10,}")
print(f"  (b) summarization @160k def: {sum(b_default):>10,}")
print(f"  (b) summarization @30k aggr: {sum(b_aggr):>10,}")
print(f"  (c) calc-subagent (+replay): {sum(c):>10,}")
print(f"  (c) calc-subagent + ckpt   : {sum(c_ckpt):>10,}")
print()
print(f"  reduction (c) vs (a): {100*(1-sum(c)/sum(a)):.0f}%   (c)+ckpt vs (a): {100*(1-sum(c_ckpt)/sum(a)):.0f}%")
print(f"  reduction (b@30k) vs (a): {100*(1-sum(b_aggr)/sum(a)):.0f}%")
print(f"  NOTE default summarizer trigger 0.8*{MODEL_WINDOW:,} = {int(0.8*MODEL_WINDOW):,};"
      f" max single-turn input (a) = {max(a):,} -> default summarization "
      f"{'NEVER FIRES' if max(a) < 0.8*MODEL_WINDOW else 'fires'} in 10 turns.")
