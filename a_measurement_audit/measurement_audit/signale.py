from common import *
import gc,warnings
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(RELEASE))
from werkzeuge.varianten.adapter import prepare_record,context,release_import
from vcg_biomarker.core.delineation.neurokit_delineator import NeuroKitDelineator
ref=pd.read_parquet(OUT/'referenz_privat.parquet');pids=set(pd.read_parquet(OUT/'p_auswahl_privat.parquet',columns=['study_id']).study_id)
cache=OUT/'signal_cache';cache.mkdir(exist_ok=True);pout=OUT/'p_ende';pout.mkdir(exist_ok=True)
def turn(a,plane):
    if a is None or len(a)<4:return None
    i,j,sx,sy={'frontal':(0,1,1,1),'transversal':(0,2,1,-1),'rechts_sagittal':(2,1,-1,1)}[plane]
    xy=a[:,[i,j]]*np.array([sx,sy]);xy-=xy[0];cross=xy[:-1,0]*xy[1:,1]-xy[1:,0]*xy[:-1,1]
    pos=cross[cross>0].sum();neg=-cross[cross<0].sum();tot=pos+neg
    if tot<1e-6:return None
    sv=np.linalg.svd(xy-xy.mean(0),compute_uv=False)
    return {'minor':float(min(pos,neg)/tot),'majority':'tie' if abs(pos-neg)<1e-12 else 'cw' if pos>neg else 'ccw','total':float(tot),'width_ratio':float(sv[1]/sv[0]) if sv[0]>0 else 0}
def sample(b,k):
    p=getattr(b,k,None);return int(p.sample) if p is not None else None
limit=int(sys.argv[1]) if len(sys.argv)>1 else len(ref)
start=time.time()
for k,row in enumerate(ref.to_dict('records')[:limit]):
    dst=cache/(str(row['study_id'])+'.json')
    if dst.exists():continue
    guard('signal_'+str(k));result={'subject_id':int(row['subject_id']),'study_id':int(row['study_id']),'age':int(row['age']),'split':row['split'],'p_stratum':row['p_stratum'],'p_duration_cache':row['p_duration_ms']}
    try:
        p=prepare_record(row);ctx=context(p);b=p['delineation'].representative_beat
        po,pp,pf,qo=[sample(b,x) for x in ['p_onset','p_peak','p_offset','r_onset']]
        result.update(status='ok',median_status=p['diagnostics'].get('status'),p_duration_new=(pf-po)*2 if po is not None and pf is not None else None,p_onset=po,p_peak=pp,p_offset=pf,qrs_onset=qo,origin=ctx.vcg_origin_assessment.provenance_note())
        for seg in ['qrs','t']:
            a=ctx.vcg_segment_origin_corrected(seg)
            for plane in ['frontal','transversal','rechts_sagittal']:result[seg+'_'+plane]=turn(a,plane)
        if int(row['study_id']) in pids:
            peaks=release_import().detect_r_peaks_multilead(p['paths']['delineation'],p['fs'])
            nk=NeuroKitDelineator();singles={}
            for lead in [1,7]:
                e=p['paths']['delineation'].copy();e[:,1]=e[:,lead]
                bb=nk.delineate(e,peaks,int(p['fs']))
                rb=min(bb,key=lambda x:abs(x.r_peak.sample-b.r_peak.sample)) if bb else None
                aa,zz=(sample(rb,'p_onset'),sample(rb,'p_offset')) if rb else (None,None)
                singles['II' if lead==1 else 'V2']={'on':aa,'off':zz,'duration':(zz-aa)*2 if aa is not None and zz is not None else None}
            result['single_lead']=singles
            center=qo if qo is not None else b.r_peak.sample
            lo=max(0,center-250);hi=min(len(p['robust']),center+75)
            np.savez_compressed(pout/(str(row['study_id'])+'_privat.npz'),ecg=p['robust'][lo:hi],raw=p['loaded'].signal_millivolts[lo:hi],time_ms=(np.arange(lo,hi)-center)*2,p_bounds=np.array([po or -1,pp or -1,pf or -1,center])-lo)
        del p,ctx,b
    except Exception as e:result.update(status='failed',reason=type(e).__name__+':'+str(e))
    jsave(dst,result);gc.collect()
    if (k+1)%10==0 or k==0:print('SIGNALS',k+1,'of',limit,'seconds',round(time.time()-start),'RSS',peak,flush=True)
    jsave(OUT/'signal_progress.json',{'processed':len(list(cache.glob('*.json'))),'target':len(ref),'rss_peak':peak,'seconds':time.time()-start})
bind([ROOT/'werkzeuge/messaudit/signale.py',ROOT/'werkzeuge/varianten/adapter.py',OUT/'referenz_privat.parquet',OUT/'p_auswahl_privat.parquet',RELEASE/'vcg_biomarker/core/delineation/fusion_delineator.py',RELEASE/'vcg_biomarker/core/delineation/neurokit_delineator.py',RELEASE/'vcg_biomarker/core/delineation/multilead_delineator.py'],'signale')
