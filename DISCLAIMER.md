# Professional-use disclaimer

**GeotechStaffEngineer is an analysis and research aid — a multiplier for a
qualified engineer's judgment, not a replacement for it, and not a design
deliverable.**

Geotechnical engineering is the practice of building on and in the ground, where
the material is the earth itself: heterogeneous, layered, partly saturated, and
sampled at only a handful of points across a site. Because the ground is variable
and only partly known, a single number is never the answer. This toolkit exists
to help an engineer run the industry-standard methods repeatedly across plausible
subsurface and loading conditions and understand the *range and spread* of
answers. It does not know your site, and it cannot exercise engineering judgment.

## What this software is — and is not

- **It is** a library of deterministic methods, a probabilistic variability
  engine, digitized references, and an LLM agent that drives them.
- **It is not** a stamped design, a professional opinion, or a substitute for
  site-specific investigation, characterization, and review.
- Using this software **creates no engineer-of-record relationship** and no
  professional engagement with the authors or contributors. No one associated
  with this project is your engineer of record.

## Your responsibilities as a user

- **A qualified, licensed professional engineer familiar with the site must
  independently review every input, assumption, method selection, and result
  before it is relied upon.** Outputs are candidate calculations to be checked,
  not conclusions to be adopted.
- Confirm that the **method applies to your problem.** Every method has
  applicability limits, sign conventions, and edge cases — these are documented
  in each module's `DESIGN.md` (and, where present, `VALIDATION.md`). Read them.
- Confirm the **inputs and units.** All quantities in this toolkit are **SI**
  (meters, kPa, kN, kN/m, degrees). Mixed or mis-scaled units are the user's
  responsibility to catch.
- Treat **LLM-agent output with particular care.** The agent can select the wrong
  method, mis-transcribe a value, or read a chart imperfectly. Its answers are a
  starting point for review, never a final basis for design.

## Validation scope

Selected modules are validated against published worked examples and benchmarks
(e.g. Fredlund & Krahn 1977, ACADS, Duncan, Griffiths & Lane, the Prandtl
solution, and GEC/Caltrans/FLAC examples). The specific problems checked, their
targets, and the pass/convention verdicts are recorded in
`validation_examples/RESULTS.md` and in the per-module `VALIDATION.md` files.
**Validation covers only those documented cases.** Agreement on a benchmark does
not certify correctness for any other geometry, parameter range, or loading
condition, and the absence of a validation entry means a result has not been
independently checked here.

## No warranty

This software is provided under the MIT License **"AS IS", WITHOUT WARRANTY OF
ANY KIND**, express or implied, including but not limited to the warranties of
merchantability, fitness for a particular purpose, and noninfringement. In no
event shall the authors or copyright holders be liable for any claim, damages, or
other liability arising from, out of, or in connection with the software or its
use. See [`LICENSE`](LICENSE) for the full terms.

Nothing in this software should be construed as engineering advice for a specific
project. Responsibility for the safety, adequacy, and code-compliance of any
design remains entirely with the licensed professional engineer who adopts it.
