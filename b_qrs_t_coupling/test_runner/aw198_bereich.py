"""Addendum: corrected OR over the pre-specified observed range (pre-registered protocol).

Aggregates only (point.json, boot_*.npz); no individual data. Fine grid R_A 0.590-0.700, R_F 0.800-0.980, rho 0.260-0.330
for the point estimate; bootstrap over the reporting grid points R_A 0.60/0.65/0.70, R_F 0.80-0.95, rho 0.268/0.3/0.329.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import json,numpy as np
OUT=(str(data_path("b.aw198")) + "/")
P=json.load(open(OUT+'point.json',encoding='utf8'))
RA=np.round(np.arange(590,701)/1000,3);RFb=np.round(np.arange(800,981,5)/1000,3);RHOb=np.round(np.arange(260,331,1)/1000,3)
RAr=np.round(np.arange(30,101,5)/100,2);RF=np.round(np.arange(50,101,5)/100,2);RHO=np.array([-.4,-.3,-.2,-.1,0.,.1,.2,.268,.3,.329,.4,.5,.6,.7,.8])
out={}
for fold,p in P.items():
    S=np.array(p['S']);V=np.array(p['V']);u=S@np.array(p['b'])
    A=RA[None,None,:];Fv=RFb[:,None,None];R=RHOb[None,:,None]
    eA=(1-A)*V[0,0];eF=(1-Fv)*V[1,1];eAF=R*np.sqrt(eA*eF)
    T11,T22,T12=np.broadcast_arrays(S[0,0]-eA,S[1,1]-eF,S[0,1]-eAF)
    det=T11*T22-T12**2;lam=(T11+T22-np.sqrt((T11-T22)**2+4*T12**2))/2;ok=lam>1e-10
    cF=np.where(ok,(-T12*u[0]+T11*u[1])/det,np.nan);i=np.unravel_index(np.nanargmin(cF),cF.shape)
    bz=np.load(OUT+'boot_'+fold+'.npz')
    ia=[list(RAr).index(x) for x in (0.6,0.65,0.7)];iF=[list(RF).index(x) for x in (0.8,0.85,0.9,0.95)];iR=[list(RHO).index(x) for x in (.268,.3,.329)]
    g=bz['grid'][:,iF][:,:,iR][:,:,:,ia].astype(float);mins=np.nanmin(g.reshape(len(g),-1),axis=1)
    out[fold]={'fine_point_min_or':float(np.exp(np.nanmin(cF))),'fine_point_min_at_RF_rho_RA':[float(RFb[i[0]]),float(RHOb[i[1]]),float(RA[i[2]])],
               'fine_point_max_or':float(np.exp(np.nanmax(cF))),'fine_inadmissible_share':float((~ok).mean()),
               'boot_box_min_or_quantiles_2.5_50_97.5':np.exp(np.quantile(mins,[.025,.5,.975])).tolist(),
               'boot_draws_all_points_admissible':int(np.isfinite(g).all(axis=(1,2,3)).sum()),'boot_n':int(len(mins))}
json.dump(out,open(OUT+'beobachteter_bereich.json','w'),indent=1);print(json.dumps(out,indent=1))
