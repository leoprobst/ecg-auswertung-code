from common import *
pout=OUT/'p_ende';sample=pd.read_csv(pout/'adjudikation_privat.csv');ref=pd.read_parquet(OUT/'referenz_privat.parquet',columns=['subject_id','p_stratum'])
decisions=json.loads((pout/'bildentscheidungen.json').read_text())
for key,d in decisions.items():
    mask=sample.audit_nr==int(key)
    for col,value in d.items():sample.loc[mask,col]=value
assert set(decisions)==set(map(str,sample.audit_nr))
assert sample.primary_category.notna().all()
sample.to_csv(pout/'adjudikation_privat.csv',index=False)
aggregate=[];rng=np.random.default_rng(47009);pop=ref.p_stratum.value_counts().to_dict();total_available=sum(pop.get(s,0) for s in ['kurz','mittel','lang']);low=high=0.;bl=np.zeros(1000);bu=np.zeros(1000)
for stratum in ['kurz','mittel','lang']:
    a=sample[sample.stratum==stratum];n=len(a);err=a.error.eq(True).to_numpy();unknown=a.error.isna().to_numpy();idx=rng.integers(0,n,(1000,n));w=pop[stratum]/total_available
    low+=w*err.mean();high+=w*(err|unknown).mean();bl+=w*err[idx].mean(1);bu+=w*(err|unknown)[idx].mean(1)
    for category in ['P_Ende_in_PR','P_Ende_in_QRS','Flutter_Artefakt','biphasische_P','Grundlinie_P_Beginn','Rauschen','korrekt_lang','technisch_plausibel','unentscheidbar']:
        ncat=int(a.primary_category.eq(category).sum());aggregate.append({'stratum':stratum,'category':category,'n_selected':n,'n_category':ncat,'fraction':float(ncat/n),'n_population':pop[stratum]})
    aggregate.append({'stratum':stratum,'category':'fehlergrenzen','n_selected':n,'n_population':pop[stratum],'confirmed_error_fraction':float(err.mean()),'unresolved_fraction':float(unknown.mean()),'error_upper_bound':float((err|unknown).mean()),'error_low_ci':ci(err[idx].mean(1)),'error_high_ci':ci((err|unknown)[idx].mean(1))})
jsave(pout/'kategorien_aggregate.json',aggregate)
jsave(pout/'fehlerquote.json',{'n_selected':len(sample),'n_population_all':len(ref),'n_population_p_available':total_available,'n_missing_p':pop.get('fehlend',0),'population_weighted_confirmed_error':low,'population_weighted_error_upper_bound':high,'confirmed_error_ci':ci(bl),'error_upper_bound_ci':ci(bu),'whole_population_error_lower':low*total_available/len(ref),'whole_population_error_upper':(high*total_available+pop.get('fehlend',0))/len(ref),'note':'stratified person-level bootstrap; the error rate refers to the evaluated boundaries and is not a physician gold standard; missing P waves reported separately'})
singles=[]
for agegroup in ['alle','18_59','60plus']:
    tmp=sample.merge(pd.read_parquet(OUT/'referenz_privat.parquet',columns=['study_id','age']),on='study_id')
    if agegroup=='18_59':tmp=tmp[tmp.age.between(18,59)]
    if agegroup=='60plus':tmp=tmp[tmp.age>=60]
    for col in ['cache_duration','new_duration','single_II','single_V2']:
        v=tmp[col].dropna().to_numpy();singles.append({'agegroup':agegroup,'measure':col,'n':len(v),'median':float(np.median(v)) if len(v) else np.nan,'p97_5':float(np.quantile(v,.975)) if len(v) else np.nan,'gt120':float((v>120).mean()) if len(v) else np.nan,'gt130':float((v>130).mean()) if len(v) else np.nan,'note':'stratified audit sample; these raw proportions are not a population norm'})
jsave(pout/'single_lead_aggregate.json',singles)
endrows=[]
for stratum in ['kurz','mittel','lang']:
    a=sample[sample.stratum==stratum]
    for kind in ['end_error','onset_error']:
        endrows.append({'stratum':stratum,'boundary':kind,'n':len(a),'confirmed_error':int(a[kind].eq(True).sum()),'no_gross_error':int(a[kind].eq(False).sum()),'unresolved':int(a[kind].isna().sum()),'basis':'12lead' if stratum=='lang' else 'II/V2'})
jsave(pout/'grenztypen_aggregate.json',endrows)
endbounds=[]
for kind in ['end_error','onset_error']:
    lower=upper=0.;bl=np.zeros(1000);bu=np.zeros(1000);rng=np.random.default_rng(47009)
    for st in ['kurz','mittel','lang']:
        a=sample[sample.stratum==st];e=a[kind].eq(True).to_numpy();u=a[kind].isna().to_numpy();w=pop[st]/total_available;idx=rng.integers(0,len(a),(1000,len(a)))
        lower+=w*e.mean();upper+=w*(e|u).mean();bl+=w*e[idx].mean(1);bu+=w*(e|u)[idx].mean(1)
    endbounds.append({'boundary':kind,'confirmed_error_lower':lower,'unresolved_included_upper':upper,'lower_ci':ci(bl),'upper_ci':ci(bu),'n_population':total_available,'note':'P onset in the short/medium strata was not adjudicated separately; its upper bound is therefore deliberately wide'})
jsave(pout/'grenzquoten.json',endbounds)
def wquant(v,w,q):
    o=np.argsort(v);v=v[o];w=w[o];s=np.cumsum(w)
    return float(v[min(np.searchsorted(s,q*s[-1]),len(v)-1)]) if len(v) and s[-1]>0 else np.nan
weighted=[];refage=pd.read_parquet(OUT/'referenz_privat.parquet',columns=['study_id','age','p_stratum']);tmp=sample.merge(refage[['study_id','age']],on='study_id')
for ag in ['alle','18_59','60plus']:
    ar=refage if ag=='alle' else refage[refage.age.between(18,59)] if ag=='18_59' else refage[refage.age>=60]
    a=tmp if ag=='alle' else tmp[tmp.age.between(18,59)] if ag=='18_59' else tmp[tmp.age>=60]
    for col in ['cache_duration','new_duration','single_II','single_V2']:
        parts=[]
        for st in ['kurz','mittel','lang']:
            x=a[a.stratum==st];parts.append((x,len(ar[ar.p_stratum==st])/len(x) if len(x) else 0.))
        def estimates(mult):
            vals=[];weights=[]
            for (x,w),m in zip(parts,mult):
                good=np.isfinite(x[col].to_numpy(dtype=float));vals.extend(x[col].to_numpy(dtype=float)[good]);weights.extend((m*w)[good])
            v=np.asarray(vals);w=np.asarray(weights)
            if not len(v) or w.sum()==0:return [np.nan]*4
            return [wquant(v,w,.5),wquant(v,w,.975),float(w@(v>120)/w.sum()),float(w@(v>130)/w.sum())]
        est=estimates([np.ones(len(x)) for x,w in parts]);rng=np.random.default_rng(47009);bb=[]
        for b in range(1000):bb.append(estimates([np.bincount(rng.integers(0,len(x),len(x)),minlength=len(x)) if len(x) else np.zeros(0) for x,w in parts]))
        bb=np.asarray(bb);weighted.append({'agegroup':ag,'measure':col,'n_selected_available':int(a[col].notna().sum()),'median':est[0],'p97_5':est[1],'gt120':est[2],'gt130':est[3],'median_ci':ci(bb[:,0]),'p97_5_ci':ci(bb[:,1]),'gt120_ci':ci(bb[:,2]),'gt130_ci':ci(bb[:,3]),'note':'inverse stratum selection weights; only the reference with P cache available; same detector family'})
jsave(pout/'single_lead_gewichtet.json',weighted)
import shutil
for cat,a in sample.groupby('primary_category'):
    r=a.iloc[0];nr=int(r.audit_nr);src=pout/(f'lang_{nr:03d}.png' if r.stratum=='lang' else f'atlas_{(nr-1)//12+1:02d}.png');shutil.copyfile(src,pout/('kategorie_'+cat+'.png'))
bind([pout/'bildentscheidungen.json',pout/'adjudikation_privat.csv',OUT/'p_auswahl_privat.parquet',ROOT/'werkzeuge/messaudit/p_ergebnis.py'],'p_ergebnis')
print((pout/'fehlerquote.json').read_text())
