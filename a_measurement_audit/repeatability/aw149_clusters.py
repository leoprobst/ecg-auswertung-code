"""Exact pairwise Spearman clusters in the existing training-partition catalog sample."""
from aw149_common import *
from scipy.stats import rankdata
import pyarrow.dataset as ds, gc
def main():
    sample=read(BASE/'varianten/AW023b/stichprobe/liste.parquet',['subject_id','study_id','split'])
    sample=sample[sample.split.eq('train')].sort_values(['subject_id','study_id'])
    assert len(sample)==2824 and sample.subject_id.is_unique
    catalog=pd.read_csv(bind(ROOT/'nachrechnung/transform_katalog.csv'));names=catalog.marker_id.tolist()
    p=next((RUN/'analysis_feature_cache').glob('*/split=train/features.parquet'));pf=pq.ParquetFile(p);bind(p)
    present=[n for n in names if n in pf.schema_arrow.names];assert len(present)==2005
    seal={'source':sha(p),'catalog':sha(ROOT/'nachrechnung/transform_katalog.csv'),'sample':sha(BASE/'varianten/AW023b/stichprobe/liste.parquet'),'code':sha(__file__)}
    raw=OUT/'cluster_train_values_privat.npy';lock=OUT/'cluster_values_binding.json'
    if not (raw.exists() and lock.exists() and json.loads(lock.read_text())==seal):
        values=np.empty((len(sample),len(names)))
        for lo in range(0,len(names),32):
            guard();cols=names[lo:lo+32]
            a=ds.dataset(str(p),format='parquet').to_table(columns=['subject_id','study_id']+cols,filter=ds.field('study_id').isin(sample.study_id.tolist())).to_pandas()
            assert not a.duplicated(['subject_id','study_id']).any()
            a=a.set_index(['subject_id','study_id']).reindex(pd.MultiIndex.from_frame(sample[['subject_id','study_id']]))
            values[:,lo:lo+len(cols)]=a[cols].to_numpy(float);del a
            print('PROJECT',lo+len(cols),len(names),flush=True)
        np.save(raw,values);dump(lock,seal)
    else:values=np.load(raw)
    groups={}
    for j in range(len(names)):
        mask=np.isfinite(values[:,j]);key=np.packbits(mask).tobytes();groups.setdefault(key,[]).append(j)
    groups=list(groups.values());groups.sort(key=lambda g:g[0]);print('AVAILABILITY_GROUPS',len(groups),flush=True)
    edges=[];comparisons=0;nonestimable=0
    cpdir=OUT/'cluster_checkpoints';cpdir.mkdir(exist_ok=True)
    for gi,left in enumerate(groups):
        out=cpdir/f'{gi:04d}.csv';sealfile=cpdir/f'{gi:04d}.json'
        if out.exists() and sealfile.exists() and json.loads(sealfile.read_text()).get('seal')==seal:
            edges.extend(pd.read_csv(out).to_dict('records'));meta=json.loads(sealfile.read_text());comparisons+=meta['comparisons'];nonestimable+=meta['nonestimable'];continue
        batch=[];count=0;missing=0
        for gj in range(gi,len(groups)):
            right=groups[gj];mask=np.isfinite(values[:,left[0]])&np.isfinite(values[:,right[0]]);n=int(mask.sum())
            pairs=len(left)*len(right) if gi!=gj else len(left)*(len(left)-1)//2
            if n<100:missing+=pairs;continue
            x=values[mask][:,left];y=values[mask][:,right]
            rx=rankdata(x,axis=0);ry=rx if gi==gj else rankdata(y,axis=0)
            rx=rx-rx.mean(0);ry=ry-ry.mean(0);sx=np.sqrt((rx*rx).sum(0));sy=np.sqrt((ry*ry).sum(0))
            with np.errstate(invalid='ignore',divide='ignore'):rho=(rx/sx).T@(ry/sy)
            if gi==gj:valid=np.triu(np.ones(rho.shape,bool),1)
            else:valid=np.ones(rho.shape,bool)
            count+=int((valid&np.isfinite(rho)).sum());missing+=int((valid&~np.isfinite(rho)).sum())
            for ii,jj in zip(*np.where(valid&(np.abs(rho)>.999))):
                batch.append({'marker_a':names[left[ii]],'marker_b':names[right[jj]],'spearman_rho':float(rho[ii,jj]),'n_train_persons_complete':n,'exact_values_equal':bool(np.array_equal(x[:,ii],y[:,jj])),'scope':'fixed train catalog sample; original feature values; pairwise complete; n>=100'})
        pd.DataFrame(batch,columns=['marker_a','marker_b','spearman_rho','n_train_persons_complete','exact_values_equal','scope']).to_csv(out,index=False)
        dump(sealfile,{'seal':seal,'comparisons':count,'nonestimable':missing});edges.extend(batch);comparisons+=count;nonestimable+=missing
        if gi%10==0:print('CLUSTER_GROUP',gi+1,len(groups),'EDGES',len(edges),flush=True)
        gc.collect()
    assert comparisons+nonestimable==2005*2004//2
    parent={n:n for n in names}
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    for r in edges:
        a=find(r['marker_a']);b=find(r['marker_b']);parent[max(a,b)]=min(a,b)
    membership={}
    for n in names:membership.setdefault(find(n),[]).append(n)
    rows=[]
    for cl,members in sorted(membership.items()):
        representative=min(members)
        for n in sorted(members):rows.append({'marker_id':n,'cluster_id':representative,'representative':representative,'cluster_size':len(members),'is_representative':n==representative,'rule':'connected components; abs(pairwise complete Spearman)>0.999; n>=100; lexicographic representative','scope':'fixed train sample n=2824; empirical redundancy, not proven algebraic equivalence'})
    pd.DataFrame(rows).to_csv(OUT/'cluster_membership.csv',index=False)
    pd.DataFrame(edges).to_csv(OUT/'cluster_edges.csv',index=False)
    dump(OUT/'cluster_summary.json',{'n_train_persons':len(sample),'markers':len(names),'availability_groups':len(groups),'estimable_pairwise_comparisons':comparisons,'nonestimable_pairwise_comparisons':nonestimable,'edges':len(edges),'clusters':len(membership),'nontrivial_clusters':sum(len(m)>1 for m in membership.values()),'representative_selection':'lexicographic, without using reliability outcomes','scope':'fixed train catalog sample, not full train population','transitivity':'components may contain pairs with rho<=0.999; only edges meet threshold'})
    finish('clusters',[OUT/'cluster_membership.csv',OUT/'cluster_edges.csv',OUT/'cluster_summary.json']);print('CLUSTERS_DONE',flush=True)
if __name__=='__main__':main()
