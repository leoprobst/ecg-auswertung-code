"""Class shares with varying joint-bootstrap estimability denominators."""
from .common import *
from .transformation import combine,CL
def main():
    guard('class_share_denominators')
    frame=pd.read_csv(ROOT/'nachrechnung/transform_katalog.csv');names=frame.marker_id.tolist();boot=[];point=[]
    for n in names:
        v=np.load(OUT/'transform'/(n+'.npz'));boot.append(v['boot']);point.append(v['point'])
    bb=np.stack(boot,axis=-1);pp=np.stack(point,axis=-1);combined=combine(pp[:,0],pp[:,1]);orig=frame['class'].map({v:i for i,v in enumerate(CL)}).to_numpy();dependent=~np.isin(orig,[0,4,5])
    path=OUT/'transform_class_counts.csv';prior=OUT/'transform_class_counts_vor_anteilsintervallen.csv'
    if not prior.exists():prior.write_bytes(path.read_bytes())
    counts=pd.read_csv(path)
    for i,row in counts.iterrows():
        norm=0 if row.normalization=='Kors_SD' else 1;mask=np.ones(len(names),bool) if row.universe=='katalog' else dependent;cl=CL.index(row['class'])
        denpoint=int((combined[norm,mask]<3).sum());counts.at[i,'n_markers_universe']=int(mask.sum());counts.at[i,'n_estimable_point']=denpoint
        numerator=(bb[norm][:,mask]==cl).sum(1);den=(bb[norm][:,mask]<3).sum(1)
        counts.at[i,'fraction_universe_point']=row.point/mask.sum();counts.at[i,'fraction_universe_ci_low'],counts.at[i,'fraction_universe_ci_high']=ci(numerator/mask.sum())
        if cl<3:
            counts.at[i,'fraction_estimable_point']=row.point/denpoint
            counts.at[i,'fraction_estimable_ci_low'],counts.at[i,'fraction_estimable_ci_high']=ci(numerator/den)
    counts.to_csv(path,index=False)
    old=json.loads((OUT/'transform_manifest.json').read_text());assert old['outputs'][str(path)]['sha256']==sha(prior)
    bind('class_shares',[ROOT/'nachrechnung/transform_katalog.csv',ROOT/'werkzeuge/rv_a1/class_proportions.py'],[path])
    dump(OUT/'transform_manifest_lineage.json',{'historical_manifest':str(OUT/'transform_manifest.json'),'historical_class_count_output_alias':str(prior),'historical_output_sha256':sha(prior),'final_class_count_output_sha256':sha(path),'authoritative_final_manifest':str(OUT/'class_shares_manifest.json'),'count_points_and_count_intervals_unchanged':True,'added':'direct bootstrap fraction intervals with each draw denominator, plus fixed universe shares'})
    print('Klassenanteile mit variablen Bootstrapnennern geprueft',flush=True)
if __name__=='__main__':main()
