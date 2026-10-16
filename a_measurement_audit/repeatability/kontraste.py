"""Matched marker estimability in each common person-bootstrap draw."""
from .common import *
def paired_statistics(x,y):
    ok=np.isfinite(x)&np.isfinite(y);n=ok.sum(1);xx=np.where(ok,x,np.nan);yy=np.where(ok,y,np.nan)
    return n,(np.nanmedian(yy,axis=1)-np.nanmedian(xx,axis=1),np.nanmedian(yy-xx,axis=1),((yy>=.75).sum(1)-(xx>=.75).sum(1))/n)
def main():
    guard('paired_contrast_estimability')
    parts=sorted((OUT/'repeat').glob('*.npz'));names=[];bags=[]
    for p in parts:
        v=np.load(p);names.extend(v['names'].tolist());bags.append(v['boot'])
    bb=np.concatenate(bags,axis=2);del bags
    frame=pd.read_csv(OUT/'repeat_marker_metrics.csv');groups=['gesamt','ohne_zeitgleich','proxy','gesamt_proxy_personen','gesamt_ein_paar','proxy_ein_paar']
    pivot=frame.pivot(index='marker_id',columns='population',values='icc').reindex(names);rows=[]
    for g in ['gesamt','gesamt_proxy_personen']:
        gi=groups.index(g);pi=groups.index('proxy');a=pivot[g].to_numpy();b=pivot.proxy.to_numpy();mask=np.isfinite(a)&np.isfinite(b)
        x=bb[gi,:,mask].T;y=bb[pi,:,mask].T;n,values=paired_statistics(x,y)
        quantities=[('difference_of_medians',np.median(b[mask])-np.median(a[mask]),values[0]),('median_of_marker_differences',np.median(b[mask]-a[mask]),values[1]),('difference_reliable_fraction',np.mean(b[mask]>=.75)-np.mean(a[mask]>=.75),values[2])]
        for metric,point,boot in quantities:
            lo,hi=ci(boot);rows.append({'comparison':'proxy_minus_'+g,'metric':metric,'n_same_markers':int(mask.sum()),'bootstrap_same_markers_min':int(n.min()),'bootstrap_same_markers_max':int(n.max()),'bootstrap_same_markers_median':float(np.median(n)),'point':float(point),'ci_low':lo,'ci_high':hi,'bootstrap_finite':int(np.isfinite(boot).sum()),'paired_estimability_each_draw':True})
    p=OUT/'repeat_contrasts.csv'
    prior=OUT/'repeat_contrasts_vor_gemeinsamer_schaetzbarkeit.csv'
    if not prior.exists():prior.write_bytes(p.read_bytes())
    pd.DataFrame(rows).to_csv(p,index=False)
    bind('paired_contrasts',[*parts,OUT/'repeat_marker_metrics.csv',ROOT/'werkzeuge/rv_a1/kontraste.py'],[p]);print('Common marker estimability per draw checked',flush=True)
    m=json.loads((OUT/'repeat_manifest.json').read_text());assert m['outputs'][str(p)]['sha256']==sha(prior)
    dump(OUT/'repeat_contrast_manifest_lineage.json',{'historical_manifest':str(OUT/'repeat_manifest.json'),'historical_manifest_sha256':sha(OUT/'repeat_manifest.json'),'historical_contrast_output_alias':str(prior),'historical_contrast_hash':sha(prior),'authoritative_final_manifest':str(OUT/'paired_contrasts_manifest.json'),'final_contrast_sha256':sha(p),'point_estimands_unchanged':True,'corrected':'same finite marker subset in each joint draw, no unreliable imputation of undefined ICC'})
if __name__=='__main__':main()
