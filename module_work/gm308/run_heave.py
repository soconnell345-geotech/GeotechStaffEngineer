import sys,time,json
sys.path.insert(0,'.')
import numpy as np
from fem2d.analysis import analyze_footing_capacity as f, bearing_capacity_factors
phi=float(sys.argv[1]); hwB=float(sys.argv[2]); mi=int(sys.argv[3]); q=float(sys.argv[4]); N=int(sys.argv[5]); fac=float(sys.argv[6])
B=2.0; nx=int(round(8*hwB))
Nc,Nq,Ng=bearing_capacity_factors(phi); est=5*Nc+q*Nq+0.5*18*B*Ng
t=time.time()
r=f(B=B,c=5.0,phi=phi,gamma=18.0,E=1e5,nu=0.3,psi=0.0,surcharge=q,domain_half_width=hwB*B,domain_depth=5*B,nx=nx,ny=20,max_iter=mi,n_load_steps=N,q_max=fac*est)
nd=r.nodes; u=r.displacements; uy=u[1::2]
s=np.abs(nd[:,1])<1e-9; xs=nd[s,0]; us=uy[s]; o=np.argsort(xs); xs,us=xs[o],us[o]
right=xs>B/2; heave=us[right].max()
xr=xs[right]; ur=us[right]
ext=xr[ur>0.05*heave].max() if heave>0 else None
print(json.dumps(dict(phi=phi,hwB=hwB,max_iter=mi,q_ult=round(r.q_ult_kPa,1),max_heave_mm=round(1e3*heave,2),
  heave_extent_B=round(ext/B,2) if ext is not None else None, side_uy_over_heave=round(float(ur[-1]/heave),3),secs=round(time.time()-t,1))),flush=True)
