# pynite_agent — design notes

Wraps `PyNiteFEA` (MIT, JWock82) for linear-elastic 2D/3D frame analysis.

## Units

Toolkit SI: m, kN, kN/m, kPa (E, G), m^2/m^4 sections; outputs kN,
kN*m, m (deflections also echoed in mm). PyNite is unit-agnostic — the
consistent kN/m/kPa set converts nothing.

## Modeling

- One material + one section object generated per member (allows fully
  heterogeneous frames from flat dicts). G defaults to E/(2(1+nu)),
  nu = 0.3; Iy defaults to Iz; J to Iy+Iz (irrelevant for planar loads).
- Supports: presets (`fixed`, `pinned`, `roller_y`, `roller_x`) or
  explicit dx..rz boolean flags.
- **2D auto-stabilization**: when every node has z = 0, out-of-plane DOFs
  (DZ, RX, RY) are restrained at all nodes so plane frames are stable
  without 3D bookkeeping (`auto_stabilize_2d=False` for true 3D work with
  z=0 geometry).
- Loads: global-direction nodal loads, member distributed (uniform or
  trapezoidal, optional x1/x2 extent) and member point loads. Downward =
  negative FY at the frame API; the continuous-beam wrapper takes
  downward-positive magnitudes and applies the sign.

## Sign conventions

Frame API reports PyNite local-axis signs plus `*_abs` magnitudes
(a -FY UDL on a simple beam yields NEGATIVE local Mz at midspan). The
continuous-beam wrapper CONVERTS to the engineering convention: sagging
positive, hogging negative (support moments), so multi-span results read
like a textbook.

## Validation anchors (in tests)

- Simply supported beam, L=6 m, w=10 kN/m, EI = 200e6*1e-4:
  R = wL/2 = 30 kN, |M|max = wL^2/8 = 45 kN*m exact;
  delta = 5wL^4/384EI = 8.4375 mm (FE gives 8.4365, 0.01%).
- Two equal 5 m spans, w=10 kN/m: reactions 0.375wL / 1.25wL / 0.375wL
  = 18.75 / 62.5 / 18.75 kN; support moment -wL^2/8 = -31.25 kN*m;
  span sagging 9wL^2/128 = 17.578 kN*m — all exact.
- Propped cantilever, UDL: prop reaction 3wL/8, fixed-end moment wL^2/8.

## Scope

Linear-static only (PyNite's P-Delta and plates are not exposed);
`opensees` remains the nonlinear/dynamic tool.
