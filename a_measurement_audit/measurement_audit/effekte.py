import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
from statistik import *
dout=OUT/'effekte';dout.mkdir(exist_ok=True)
p=pd.read_parquet(OUT/'paare_ergaenzt_privat.parquet');groups=pd.read_parquet(OUT/'repeatability/paargruppen_privat.parquet',columns=['subject_id','first_study_id','second_study_id','stable'])
pairkeys=['subject_id','first_study_id','second_study_id']
assert not p.duplicated(pairkeys).any() and not groups.duplicated(pairkeys).any()
assert set(map(tuple,p[pairkeys].to_numpy()))==set(map(tuple,groups[pairkeys].to_numpy()))
groups=groups.set_index(pairkeys).reindex(pd.MultiIndex.from_frame(p[pairkeys])).reset_index()
assert p[pairkeys].equals(groups[pairkeys]);p['stable']=groups.stable
keys=json.loads((OUT/'befund_schluessel.json').read_text());rows=[];inputs=[]
def wmedian(values,weights):
    v=values;w=weights;cs=np.cumsum(w)
    if not len(cs) or cs[-1]<=0:return np.nan
    # Weighted replication median, with averaging of two middle order statistics.
    lo=(cs[-1]+1)//2;hi=(cs[-1]+2)//2
    return (v[np.searchsorted(cs,lo)]+v[np.searchsorted(cs,hi)])/2
for key in keys:
    bid=key['befund'];marker=key['feature_id'];src=Path(str(data_path("a.nachrechnung")))/bid/'development_projection.parquet';footer(src);inputs.append(src)
    f=pd.read_parquet(src,columns=['subject_id','study_id','split','label','label_state','eligibility_state',marker]);assert set(f.split)<={'train','validation'}
    for fold in ['train','validation']:
        a=f[f.split==fold].copy();assert a.subject_id.is_unique,'index cohort is not a person-level unit'
        avail=np.isfinite(a[marker]) & a.label.isin([0,1]);ff=a[avail];pos=ff.label.eq(1).to_numpy();v=ff[marker].to_numpy();subjects=ff.subject_id.to_numpy();diff=v[pos].mean()-v[~pos].mean();med=np.median(v[pos])-np.median(v[~pos]);op=np.argsort(v[pos]);on=np.argsort(v[~pos])
        for pop in ['gesamt','stabil','gesamt_ein_Paar','stabil_ein_Paar']:
            pp=p[(p.split==fold)&p.subject_id.isin(a.subject_id)]
            if pop.startswith('stabil'):pp=pp[pp.stable]
            if pop.endswith('ein_Paar'):pp=pp.sort_values(['subject_id','first_ecg_time','second_ecg_time']).drop_duplicates('subject_id')
            good=np.isfinite(pp[marker+'_first'])&np.isfinite(pp[marker+'_second']);pp=pp[good]
            delta=(pp[marker+'_second']-pp[marker+'_first']).to_numpy();psub=pp.subject_id.to_numpy();ids=np.union1d(a.subject_id,psub);u=len(ids);iv=np.searchsorted(ids,subjects);ip=np.searchsorted(ids,psub)
            agg=np.zeros((7,u))
            for j,vals,ind in [(0,pos.astype(float),iv),(1,(~pos).astype(float),iv),(2,v*pos,iv),(3,v*(~pos),iv),(4,np.ones(len(delta)),ip),(5,delta,ip),(6,delta*delta,ip)]:np.add.at(agg[j],ind,vals)
            rc=1.96*np.std(delta,ddof=1) if len(delta)>1 else np.nan;ratio=abs(diff)/rc if rc>0 else np.nan
            bootdiff=[];bootmed=[];bootrc=[];bootratio=[];rng=np.random.default_rng(47009)
            for b in range(1000):
                w=np.bincount(rng.integers(0,u,u),minlength=u);n1,n0,s1,s0,nd,sd,sdd=agg@w
                dd=s1/n1-s0/n0 if min(n0,n1)>0 else np.nan
                rc0=1.96*np.sqrt(max(0,(sdd-sd*sd/nd)/(nd-1))) if nd>1 else np.nan
                bootdiff.append(dd);bootrc.append(rc0);bootratio.append(abs(dd)/rc0 if rc0>0 else np.nan)
                ww=w[iv];bootmed.append(wmedian(v[pos][op],ww[pos][op])-wmedian(v[~pos][on],ww[~pos][on]))
            ns=len(np.unique(psub));r={'befund':bid,**key,'split':fold,'population':pop,'unit':'mV' if marker.endswith('_mv') else 'dimensionslos','n_index_candidates':len(a),'n_index_usable':len(ff),'n_positive':int(pos.sum()),'n_negative':int((~pos).sum()),'missing_positive':int(((a.label==1)&~avail).sum()),'missing_negative':int(((a.label==0)&~avail).sum()),'mean_difference':diff,'mean_difference_ci':ci(bootdiff),'median_difference':med,'median_difference_ci':ci(bootmed),'n_pairs':len(pp),'n_pair_subjects':ns,'repeatability_bias':float(delta.mean()) if len(delta) else np.nan,'rc':rc,'sdc95':rc,'rc_ci':ci(bootrc) if ns>=30 else [np.nan,np.nan],'ratio_abs_mean_rc':ratio,'ratio_ci':ci(bootratio) if ns>=30 else [np.nan,np.nan],'source':str(src),'filter':f'fold={fold}; label0/1; finite exact marker; {pop}; pair subjects in index cohort','script':'werkzeuge/messaudit/effekte.py'}
            lower,upper=r['ratio_ci'];r['decision']='groesser' if lower>1 else 'kleiner' if upper<1 else 'unentschieden' if np.isfinite(lower) else 'nicht_schaetzbar'
            rows.append(r)
        print('EFFECT',bid,fold,'n',len(ff),'difference',diff,flush=True)
    pd.DataFrame(rows).to_csv(dout/'kennzahlen.csv',index=False)
jsave(dout/'aggregate.json',rows);bind([*inputs,OUT/'paare_ergaenzt_privat.parquet',OUT/'repeatability/paargruppen_privat.parquet',ROOT/'werkzeuge/messaudit/effekte.py',dout/'kennzahlen.csv'],'effekte')
