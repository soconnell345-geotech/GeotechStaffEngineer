import sys,time,json
sys.path.insert(0,'.')
from fem2d.analysis import analyze_footing_capacity as f
phi=float(sys.argv[1]); hwB=float(sys.argv[2]); mi=int(sys.argv[3]); q=float(sys.argv[4]) if len(sys.argv)>4 else 0.0
N=int(sys.argv[5]) if len(sys.argv)>5 else 45
B=2.0; nx=int(round(8*hwB))
t=time.time()
kw=dict(B=B,c=5.0,phi=phi,gamma=18.0,E=1e5,nu=0.3,psi=0.0,surcharge=q,
        domain_half_width=hwB*B,domain_depth=5*B,nx=nx,ny=20,max_iter=mi,n_load_steps=N)
# common q_max from the default-width estimate so load grids match
from fem2d.analysis import bearing_capacity_factors
Nc,Nq,Ng=bearing_capacity_factors(phi); est=5*Nc+q*Nq+0.5*18*B*Ng
kw['q_max']=float(sys.argv[6])*est if len(sys.argv)>6 else 1.6*est
if len(sys.argv)>7: kw['iteration']=sys.argv[7]  # GM-308 (a)
r=f(**kw)
it=None
print(json.dumps(dict(iteration=kw.get('iteration'),phi=phi,hwB=hwB,nx=nx,max_iter=mi,q=q,N=N,qmax=round(kw["q_max"],1),bracketed=r.collapse_bracketed,q_ult=round(r.q_ult_kPa,1),nok=r.n_steps_converged,
  settled=r.n_steps_settled,maxit_warn=any('max_iter' in w for w in r.warnings),secs=round(time.time()-t,1))),flush=True)
