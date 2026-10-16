"""Denominators for finite sign comparisons and non-estimable projections."""
from .common import *
def main():
    guard('sign_estimability_denominators')
    src=OUT/'sign_continuous_privat.parquet'
    f=read(src,['population','fold','subject','metric','delta']);pointpath=OUT/'sign_continuous.csv';point=pd.read_csv(pointpath,dtype=str)
    point['n_candidate']=''
    available=[];candidates={}
    checkpoints=sorted((OUT/'sign').glob('*.json'))
    for p in checkpoints:
        r=json.loads(p.read_text());key=(r['population'],r['fold']);candidates[key]=candidates.get(key,0)+1
        for seg in ['qrs','t']:
            for plane in ['frontal','transversal','rechts_sagittal']:
                a=r.get('methods',{}).get('inverse_dower',{}).get(seg+'_'+plane);b=r.get('methods',{}).get('plus',{}).get(seg+'_'+plane)
                available.append({'population':key[0],'fold':key[1],'segment':seg,'plane':plane,'minus_available':a is not None,'plus_available':b is not None,'both_available':a is not None and b is not None,'both_unavailable':a is None and b is None,'minus_only':a is not None and b is None,'plus_only':a is None and b is not None})
    rows=[]
    for key,z in pd.DataFrame(available).groupby(['population','fold','segment','plane']):
        row=dict(zip(['population','fold','segment','plane'],key));row['n_candidate']=len(z)
        for col in ['minus_available','plus_available','both_available','both_unavailable','minus_only','plus_only']:row[col]=int(z[col].sum())
        assert row['both_available']+row['both_unavailable']+row['minus_only']+row['plus_only']==row['n_candidate'];rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'sign_availability.csv',index=False)
    grouped=f.groupby(['population','fold','metric'])
    for i,row in point.iterrows():
        key=(row.population,row.fold,row.metric);z=grouped.get_group(key);valid=z[np.isfinite(z.delta)]
        assert valid.subject.is_unique and len(valid)==int(row.n_pairs)
        point.at[i,'n_subjects']=str(valid.subject.nunique());point.at[i,'n_candidate']=str(candidates[key[:2]])
    prior=OUT/'sign_continuous_vor_nennerpraezisierung.csv'
    if not prior.exists():prior.write_bytes(pointpath.read_bytes())
    point.to_csv(pointpath,index=False)
    old=json.loads((OUT/'sign_manifest.json').read_text());assert old['outputs'][str(pointpath)]['sha256']==sha(prior)
    bind('sign_estimability',[src,OUT/'sign_manifest.json',ROOT/'werkzeuge/rv_a1/sign_availability.py'],[pointpath,OUT/'sign_availability.csv'])
    dump(OUT/'sign_manifest_lineage.json',{'historical_manifest':str(OUT/'sign_manifest.json'),'historical_continuous_output_alias':str(prior),'historical_output_sha256':sha(prior),'final_continuous_sha256':sha(pointpath),'authoritative_final_manifest':str(OUT/'sign_estimability_manifest.json'),'all_numeric_estimates_CI_and_pair_counts_unchanged':True,'corrected':'subjects counted on finite paired rows; candidate count is full frozen fold; projection missingness explicit'})
    print('Sign denominators and projection missingness checked',flush=True)
if __name__=='__main__':main()
