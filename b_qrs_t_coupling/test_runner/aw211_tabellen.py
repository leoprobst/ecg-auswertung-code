"""Tables (CSV, Markdown) and figure from the aggregates. Reads only aggregates and, for the reference quantity d_CC_log, the outcome file (footer check)."""
import os,sys,json,csv
from pathlib import Path
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from werkzeuge.b_test import aw211_common as C
import numpy as np
r1=C.load(C.OUT/'t1_results.json');r2=C.load(C.OUT/'t2_results.json');pt=C.load(C.OUT/'t1_point.json')
M=r1['metrics']
def fm(k,d=2,scale=1.0):
    v=M[k]['point'];c=M[k]['ci']
    return f"{v*scale:.{d}f} ({c[0]*scale:.{d}f}; {c[1]*scale:.{d}f})" if c else f"{v*scale:.{d}f} (n/a)"
L=[]
# Table 1: LVEF groups
L.append('### T1. Repeat error by LVEF group (SD of differences, ICC; person bootstrap 95 %, 1,000 valid draws)\n')
L.append('| Set | Quantity | Group | Pairs | SD angle (deg) | ICC angle | SD Fréchet (dev. SD) | ICC Fréchet | Error correlation |')
L.append('|---|---|---|---:|---|---|---|---|---|')
for st,nm in [('SA','S_A (index = first recording)'),('SB','S_B (all pairs with reference)')]:
    for g,gn in [('low','LVEF < 40 %'),('pres','LVEF >= 40 %')]:
        L.append(f"| {nm} | {gn} | | {int(M[f'{st}_{g}_n']['point'])} | {fm(f'{st}_{g}_A_sd',1)} | {fm(f'{st}_{g}_A_icc',3)} | {fm(f'{st}_{g}_F_sd',3)} | {fm(f'{st}_{g}_F_icc',3)} | {fm(f'{st}_{g}_rho',3)} |")
L.append('')
L.append('| Set | Quantity | Angle | Fréchet raw | Fréchet log1p |')
L.append('|---|---|---|---|---|')
for st in ['SA','SB']:
    L.append(f"| {st} | SD ratio low/preserved | {fm(f'{st}_A_sdratio',3)} | {fm(f'{st}_F_sdratio',3)} | {fm(f'{st}_Fl_sdratio',3)} |")
    L.append(f"| {st} | SD ratio adjusted for amplitude quintile | {fm(f'{st}_A_qp_adj_sdratio',3)} | {fm(f'{st}_F_qp_adj_sdratio',3)} | {fm(f'{st}_Fl_qp_adj_sdratio',3)} |")
    L.append(f"| {st} | adjusted / crude | {fm(f'{st}_A_adj_over_crude',3)} | {fm(f'{st}_F_adj_over_crude',3)} | {fm(f'{st}_Fl_adj_over_crude',3)} |")
    L.append(f"| {st} | ICC difference low minus preserved | {fm(f'{st}_A_iccdiff',3)} | {fm(f'{st}_F_iccdiff',3)} | {fm(f'{st}_Fl_iccdiff',3)} |")
L.append('')
# Table 2: quintiles
L.append('### T2. Repeat error by QRS amplitude quintile (pairs, 12,905; cut points 819, 1,024, 1,241, 1,543 µV)\n')
L.append('| Quintile | Pairs | SD angle (deg) | ICC angle | SD Fréchet (dev. SD) | ICC Fréchet |')
L.append('|---|---:|---|---|---|---|')
for k in range(1,6):
    L.append(f"| Q{k} | {int(M[f'Q{k}_n']['point'])} | {fm(f'Q{k}_A_sd',1)} | {fm(f'Q{k}_A_icc',3)} | {fm(f'Q{k}_F_sd',3)} | {fm(f'Q{k}_F_icc',3)} |")
L.append('')
L.append('| Quantity | Angle | Fréchet raw | Fréchet log1p |')
L.append('|---|---|---|---|')
L.append(f"| SD ratio Q1/Q5 | {fm('amp_A_q1_over_q5_sdratio',3)} | {fm('amp_F_q1_over_q5_sdratio',3)} | {fm('amp_Fl_q1_over_q5_sdratio',3)} |")
L.append(f"| SD ratio per quintile step (trend) | {fm('amp_A_trend_sdratio_step',3)} | {fm('amp_F_trend_sdratio_step',3)} | {fm('amp_Fl_trend_sdratio_step',3)} |")
L.append('')
# Table 3: kappa
L.append('### T3. Group-specific error in the outcome case set (anchor EF_mix, kappa scenario analysis)\n')
L.append('| Quantity | train | validation |')
L.append('|---|---|---|')
def b2(fold,k,exp=False,d=3):
    v=r1['boot2'][fold]['point'][k];c=r1['boot2'][fold]['ci'][k]
    return f"{v:.{d}f} ({c[0]:.{d}f}; {c[1]:.{d}f})" if c else f"{v:.{d}f} (n/a)"
for k,lab,d in [('R_A_mix','R_A (angle, group-specific mixture)',3),('R_F_mix','R_F (Fréchet)',3),('rho_mix','Error correlation',3),('pcorr_mix','implied partial latent correlation',3),('R_A_pool','R_A with pooled error (S_B)',3),('ratio_A','Error variance ratio angle (cases/controls)',3),('ratio_F','Error variance ratio Fréchet',3),('pi','Case share',3)]:
    L.append(f"| {lab} | {b2('train',k,d=d)} | {b2('validation',k,d=d)} |")
for k,lab in [('logor_mix','corrected Fréchet OR, EF_mix'),('logor_pool','corrected Fréchet OR, EF_pool (control anchor)')]:
    L.append(f"| {lab} | {r1['boot2']['train']['point'][k]:.1f} ({r1['boot2']['train']['ci'][k][0]:.1f}; {r1['boot2']['train']['ci'][k][1]:.3g}) | {r1['boot2']['validation']['point'][k]:.1f} ({r1['boot2']['validation']['ci'][k][0]:.1f}; {r1['boot2']['validation']['ci'][k][1]:.3g}) |")
L.append(f"| admissible draws EF_mix of 1,000 | {r1['boot2']['train']['admissible_mix']} | {r1['boot2']['validation']['admissible_mix']} |")
for fold in ['train','validation']:pass
pk=lambda fold,t:r1['boot2'][fold][f'kappa_{t}']
L.append(f"| kappa at which T becomes singular (point; bootstrap 95 %) | {pt['kappa_point']['train']['kappa_1.25']:.2f} ({pk('train',1.25)['boundary_ci'][0]:.2f}; {pk('train',1.25)['boundary_ci'][1]:.2f}) | {pt['kappa_point']['validation']['kappa_1.25']:.2f} ({pk('validation',1.25)['boundary_ci'][0]:.2f}; {pk('validation',1.25)['boundary_ci'][1]:.2f}) |")
L.append(f"| Status at the point (target 1.25 and 1.0) | {pk('train',1.25)['point_status']} | {pk('validation',1.25)['point_status']} |")
L.append(f"| Draws with true crossing (1.25 / 1.0) | {pk('train',1.25)['n_kreuzung']} / {pk('train',1.0)['n_kreuzung']} | {pk('validation',1.25)['n_kreuzung']} / {pk('validation',1.0)['n_kreuzung']} |")
L.append(f"| kappa_plaus (upper limit of the variance ratio / point) | {r1['kappa_plaus']['kappa_plaus']:.2f} | {r1['kappa_plaus']['kappa_plaus']:.2f} |")
L.append('')
# Table 4: part 2 grid S1
L.append('### T4. Delta-adjusted imputation, scheme S1 (Fréchet raw), pooled OR (Rubin 95 %), attenuating direction\n')
L.append('| Path | d (SD) | train OR | train share at lower bound 0 | validation OR | validation share at lower bound 0 |')
L.append('|---|---:|---|---:|---|---:|')
gt={(g['scheme'],g['path'],g['direction'],g['d']):g for g in r2['folds']['train']['grid']};gv={(g['scheme'],g['path'],g['direction'],g['d']):g for g in r2['folds']['validation']['grid']}
for path,pn in [('P1','Cases -d'),('P2','Controls +d'),('P3','Opposing')]:
    for d in [0,0.5,1,2,3,4,8]:
        a=gt[('S1',path,'abschwaechend',d)];b=gv[('S1',path,'abschwaechend',d)]
        L.append(f"| {pn} | {d:g} | {a['or']:.2f} ({a['lo']:.2f}; {a['hi']:.2f}) | {a['floor_share']:.2f} | {b['or']:.2f} ({b['lo']:.2f}; {b['hi']:.2f}) | {b['floor_share']:.2f} |")
L.append('')
L.append('### T5. Tipping points delta* (SD of the respective scale); Monte Carlo halves in parentheses\n')
L.append('| Scheme | Path | Target | train | validation |')
L.append('|---|---|---|---|---|')
def tp(x):
    if x['d'] is None:return f"not reached up to 8 (OR at 8: {x['or_at_8']:.2f})"
    return f"{x['d']:.2f} ({x['half1']['d']:.2f}; {x['half2']['d']:.2f})" if x['half1']['d'] is not None else f"{x['d']:.2f}"
for a,b in zip(r2['folds']['train']['tips'],r2['folds']['validation']['tips']):
    L.append(f"| {a['scheme']} | {a['path']} | {a['target']} | {tp(a)} | {tp(b)} |")
L.append('')
(C.OUT/'tabellen.md').write_text('\n'.join(L),encoding='utf8')
# CSV
def wcsv(name,rows):
    with (C.OUT/name).open('w',newline='',encoding='utf8') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0].keys()));w.writeheader()
        for r in rows:w.writerow(r)
wcsv('t1_metriken.csv',[{'metrik':k,'punkt':v['point'],'ci_low':(v['ci'] or [None,None])[0],'ci_high':(v['ci'] or [None,None])[1]} for k,v in M.items()])
rows=[]
for fold in ['train','validation']:
    for g in r2['folds'][fold]['grid']:rows.append({'fold':fold,**{k:g[k] for k in ['scheme','path','direction','d','or','lo','hi','floor_share']}})
wcsv('t2_gitter.csv',rows)
rows=[]
for fold in ['train','validation']:
    for t in r2['folds'][fold]['tips']:rows.append({'fold':fold,'scheme':t['scheme'],'path':t['path'],'target':t['target'],'status':t['status'],'d':t['d'],'or_at_8':t.get('or_at_8'),'d_half1':t['half1']['d'],'d_half2':t['half2']['d']})
wcsv('t2_kipppunkte.csv',rows)
# d_CC on the log1p scale (reference quantity for S2, addition to the protocol)
f,_,scale=C.outcome_frame();g=f.loc[f.eligible_B3.eq(True)]
res={}
sdl=r2['sd_log_train']
for fold in ['train','validation']:
    x=g[g.split==fold];y=(x.preferred_lvef<40).to_numpy();lf=np.log1p(x[C.F].to_numpy(float))
    res[fold]=float((lf[y].mean()-lf[~y].mean())/sdl)
C.dump(C.OUT/'d_cc_log.json',res);print('d_CC_log',res)
# Figure
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,3,figsize=(15,4.4))
q=range(1,6)
for s,lab,c in [('A','angle (deg)','tab:blue'),('F','Fréchet (dev-SD)','tab:red')]:
    v=[M[f'Q{k}_{s}_sd']['point'] for k in q];lo=[M[f'Q{k}_{s}_sd']['ci'][0] for k in q];hi=[M[f'Q{k}_{s}_sd']['ci'][1] for k in q]
    a=ax[0] if s=='A' else ax[0].twinx()
    a.errorbar(list(q),v,yerr=[np.array(v)-lo,np.array(hi)-v],color=c,marker='o',capsize=3,label=lab);a.set_ylabel('SD of repeat differences, '+lab,color=c)
ax[0].set_xlabel('QRS amplitude quintile (1 = lowest)');ax[0].set_title('A. Repeat error by amplitude quintile')
for i,(s,lab) in enumerate([('A','angle'),('F','Fréchet')]):
    for j,st in enumerate(['SA','SB']):
        for k,(g,col) in enumerate([('low','tab:red'),('pres','tab:blue')]):
            v=M[f'{st}_{g}_{s}_sd'];x=i*3+j+(k*0.35)-0.17
            ax[1].errorbar([x],[v['point']/ (1 if s=='A' else 1)/(M[f'SB_pres_{s}_sd']['point'])],yerr=[[ (v['point']-v['ci'][0])/M[f'SB_pres_{s}_sd']['point']],[(v['ci'][1]-v['point'])/M[f'SB_pres_{s}_sd']['point']]],color=col,marker='o',capsize=3)
ax[1].axhline(1,color='gray',lw=.8,ls=':');ax[1].set_xticks([0.0,1.0,3.0,4.0]);ax[1].set_xticklabels(['angle S_A','angle S_B','Fréchet S_A','Fréchet S_B']);ax[1].set_ylabel('SD of differences relative to LVEF>=40% (S_B)');ax[1].set_title('B. Red: LVEF<40%, blue: LVEF>=40%')
for fold,ls in [('train','-'),('validation','--')]:
    for path,col,lab in [('P1','tab:green','cases down'),('P2','tab:orange','controls up'),('P3','tab:purple','both')]:
        pts=sorted([(g['d'],g['or']) for g in r2['folds'][fold]['grid'] if g['scheme']=='S1' and g['path']==path and g['direction']=='abschwaechend'])
        ax[2].plot([p[0] for p in pts],[p[1] for p in pts],color=col,ls=ls,marker='o',ms=3,label=f'{lab} ({fold})')
ax[2].axhline(1.25,color='gray',ls=':',lw=.8);ax[2].axhline(1.0,color='gray',ls=':',lw=.8);ax[2].set_xlabel('delta shift of imputed Fréchet (development SD)');ax[2].set_ylabel('pooled OR per SD');ax[2].set_title('C. Delta-adjusted MI');ax[2].legend(fontsize=6)
fig.tight_layout();fig.savefig(C.OUT/'aw211_abb.png',dpi=140);print('FIG done')
