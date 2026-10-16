from .common import *
from scipy.stats import rankdata
import itertools,gc
from werkzeuge.transform_katalog.metriken import vergleichen
CL=['stabil','bedingt','transformabhaengig','nicht_schaetzbar','konstant_identisch','nicht_pruefbar']
RULES=[(.97,.4,.08,.85,1.5,.30),(.95,.5,.10,.80,2.,.40),(.90,.75,.15,.75,2.5,.50)]
def classify(r,w,s,n,valid,rule=RULES[1]):
    a,b,c,d,e,f=rule
    return np.where(valid & (n>=100),np.where((r>=a)&(w<=b)&(s<=c),0,np.where((r<d)|(w>e)|(s>f),2,1)),3).astype('int8')
def metrics(x,y,weights):
    good=np.isfinite(x)&np.isfinite(y);x=x[good];y=y[good];weights=weights[:,good].astype(float)
    if len(x)<2:return np.full((len(weights),4),np.nan),np.zeros(len(weights)),np.zeros(len(weights),bool)
    # Shift/scale protects native large/small units; relative width and ranks invariant.
    cen=(x.mean()+y.mean())/2;scale=max(np.std(np.r_[x,y]),1e-100);x=(x-cen)/scale;y=(y-cen)/scale
    def ranks(v):
        order=np.argsort(v,kind='stable');v0=v[order];starts=np.r_[0,np.flatnonzero(v0[1:]!=v0[:-1])+1]
        counts=np.add.reduceat(weights[:,order],starts,axis=1);r=np.cumsum(counts,axis=1)-counts/2+.5
        ix=np.empty(len(order),int);ix[order]=np.repeat(np.arange(len(starts)),np.diff(np.r_[starts,len(v)]))
        return r[:,ix]
    n=weights.sum(1);safe=np.maximum(n,1)
    rx,ry=ranks(x),ranks(y);mx=(weights*rx).sum(1)/safe;my=(weights*ry).sum(1)/safe
    dx=rx-mx[:,None];dy=ry-my[:,None]
    with np.errstate(divide='ignore',invalid='ignore'):
        rho=(weights*dx*dy).sum(1)/np.sqrt((weights*dx*dx).sum(1)*(weights*dy*dy).sum(1))
        qx=np.clip(np.floor(5*(rx-.5)/safe[:,None]),0,4);qy=np.clip(np.floor(5*(ry-.5)/safe[:,None]),0,4)
        switch=(weights*(qx!=qy)).sum(1)/safe
        vx=(weights*x*x).sum(1)-(weights*x).sum(1)**2/safe
        vy=(weights*y*y).sum(1)-(weights*y).sum(1)**2/safe
        delta=y-x;vd=(weights*delta*delta).sum(1)-(weights*delta).sum(1)**2/safe
        wd=3.92*np.sqrt(np.maximum(vd,0)/vx);ws=3.92*np.sqrt(np.maximum(vd,0)/((vx+vy)/2))
    return np.array([rho,wd,ws,switch]).T,n,(vx>0)&(vy>0)&np.isfinite(rho)
def combine(a,b):return np.where((a<3)&(b<3),np.maximum(a,b),np.where((a==4)&(b==4),4,np.where((a==5)&(b==5),5,3)))
def main():
    frame=pd.read_csv(ROOT/'nachrechnung/transform_katalog.csv');names=frame.marker_id.tolist();allowed=frame.loc[frame.admitted,'marker_id'].tolist()
    files=sorted((VAR/'bloecke').glob('*.parquet'));assert len(files)==29
    cols=['subject_id','study_id','split']+[p+n for p in ('kors__','dower__') for n in allowed]
    for p in files:
        seal=json.loads(p.with_suffix('.json').read_text());assert sha(p)==seal['sha256']
    data=pd.concat([read(p,cols) for p in files],ignore_index=True);assert len(data)==5648 and data.subject_id.is_unique
    assert set(data.split)=={'train','validation'}
    masks=[data.split.eq(f).to_numpy() for f in ('train','validation')]
    weights=[]
    for i,mask in enumerate(masks):
        n=mask.sum();rng=np.random.default_rng(SEED+i);w=np.zeros((B,n),np.int16)
        for k in range(B):w[k]=np.bincount(rng.integers(0,n,n),minlength=n)
        weights.append(w)
    dst=OUT/'transform';dst.mkdir(exist_ok=True)
    point=[]
    for j,row in enumerate(frame.to_dict('records')):
        guard('transform_marker_'+str(j))
        path=dst/(row['marker_id']+'.npz')
        if path.exists():continue
        if not row['admitted']:
            np.savez_compressed(path,boot=np.full((2,B),5,dtype='int8'),point=np.full((2,3),5,dtype='int8'),metrics=np.full((3,4),np.nan));continue
        x=data['kors__'+row['marker_id']].to_numpy(float);y=data['dower__'+row['marker_id']].to_numpy(float)
        mm=[];boot=np.empty((2,B,2),np.int8)
        for f,mask in enumerate(masks):
            xx=x[mask];yy=y[mask];good=np.isfinite(xx)&np.isfinite(yy)
            constant=good.sum()>1 and np.std(xx[good])==0 and np.array_equal(xx[good],yy[good])
            m,n,valid=metrics(xx,yy,np.ones((1,len(xx))));mm.append(m[0])
            for lo in range(0,B,25):
                bm,bn,bvalid=metrics(xx,yy,weights[f][lo:lo+25])
                for norm in range(2):boot[f,lo:lo+25,norm]=4 if constant else classify(bm[:,0],bm[:,1+norm],bm[:,3],bn,bvalid)
        m,n,valid=metrics(x,y,np.ones((1,len(x))));mm.append(m[0]);mm=np.asarray(mm)
        pp=np.empty((2,3),np.int8)
        for norm in range(2):
            for f in range(3):
                m0=vergleichen(x[masks[f]] if f<2 else x,y[masks[f]] if f<2 else y)
                pp[norm,f]=4 if m0['class']=='konstant_identisch' else classify(mm[f,0],mm[f,norm+1],mm[f,3],np.array(m0['n_pairs']),np.isfinite(mm[f,0]))
        combined=np.stack([combine(boot[0,:,norm],boot[1,:,norm]) for norm in range(2)])
        np.savez_compressed(path,boot=combined,point=pp,metrics=mm)
        if j%50==0:print('TRANSFORM',j+1,len(frame),flush=True)
        gc.collect()
    bb=[];pp=[];met=[]
    for n in names:
        v=np.load(dst/(n+'.npz'));bb.append(v['boot']);pp.append(v['point']);met.append(v['metrics'])
    bb=np.stack(bb,axis=-1);pp=np.stack(pp,axis=-1);met=np.stack(met)
    combined=combine(pp[:,0],pp[:,1]);orig=frame['class'].map({v:i for i,v in enumerate(CL)}).to_numpy()
    assert np.array_equal(combined[0],orig),'Klassifikation reproduziert Original nicht'
    dependent=~np.isin(orig,[0,4,5]);den=(dependent&(orig<3));assert den.sum()==1755
    bootrows=[]
    for norm in range(2):
        for universe,mask in [('katalog',np.ones(len(names),bool)),('matrixabhaengig',dependent)]:
            for cl in range(6):
                vals=(bb[norm][:,mask]==cl).sum(1)
                bootrows.append({'normalization':['Kors_SD','pooled_SD'][norm],'universe':universe,'class':CL[cl],'point':int((combined[norm,mask]==cl).sum()),'ci_low':ci(vals)[0],'ci_high':ci(vals)[1],'bootstrap_den_estimable_min':int((bb[norm][:,mask]<3).sum(1).min()),'bootstrap_den_estimable_max':int((bb[norm][:,mask]<3).sum(1).max())})
    pd.DataFrame(bootrows).to_csv(OUT/'transform_class_counts.csv',index=False)
    ranking=pd.DataFrame({'marker_id':names,'rho_min':np.minimum(met[:,0,0],met[:,1,0]),'matrix_dependent':dependent,'original_class':frame['class']})
    ranking['p_stable_kors']=(bb[0]==0).mean(0);ranking['p_stable_pooled']=(bb[1]==0).mean(0)
    ranking['p_nonestimable']=(bb[0]>=3).mean(0);ranking.to_csv(OUT/'transform_stability.csv',index=False)
    top=ranking[den].sort_values(['rho_min','marker_id'],ascending=[False,True]).head(10);top.to_csv(OUT/'transform_top10.csv',index=False)
    grids=list(itertools.product(*[sorted(set(r[k] for r in RULES)) for k in range(6)]));gridrows=[]
    foldn=[frame[fold+'_n_pairs'].fillna(0).to_numpy(float) for fold in ['train','validation']]
    for norm in range(2):
        for rule in grids:
            cc=[]
            for f in range(2):
                c=classify(met[:,f,0],met[:,f,norm+1],met[:,f,3],foldn[f],np.isfinite(met[:,f,0]),rule)
                c[orig==4]=4;c[orig==5]=5;cc.append(c)
            c=combine(*cc);gridrows.append({'normalization':['Kors_SD','pooled_SD'][norm],'rule':str(rule),'class_switches':int((c!=orig).sum()),'stable_matrixdependent':int((c[dependent]==0).sum()),**{CL[k]:int((c==k).sum()) for k in range(6)}})
    pd.DataFrame(gridrows).to_csv(OUT/'transform_grid.csv',index=False)
    foldrows=[]
    for norm in range(2):
        for f in range(3):foldrows.append({'normalization':['Kors_SD','pooled_SD'][norm],'fold':['train','validation','pool'][f],'switches_vs_two_fold':int((pp[norm,f]!=orig).sum()),**{CL[k]:int((pp[norm,f]==k).sum()) for k in range(6)}})
    pd.DataFrame(foldrows).to_csv(OUT/'transform_fold_sensitivity.csv',index=False)
    dump(OUT/'transform_summary.json',{'n_subjects':len(data),'original_reproduced':True,'dependent_estimable':int(den.sum()),'symmetric_switches':int((combined[1]!=orig).sum()),'grid_rules':len(grids),'bootstrap_draws':B,'seed':SEED})
    bind('transform',[*files,ROOT/'nachrechnung/transform_katalog.csv'],[OUT/n for n in ['transform_class_counts.csv','transform_stability.csv','transform_top10.csv','transform_grid.csv','transform_summary.json']])
    print((OUT/'transform_summary.json').read_text(),flush=True)
if __name__=='__main__':main()
