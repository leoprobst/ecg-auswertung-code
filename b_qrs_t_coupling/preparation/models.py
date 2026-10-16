"""Train-feste logistische Fits, gepaarter Patientenbootstrap; keine Testpopulation."""
import numpy as np
from scipy.special import expit
from werkzeuge.nachrechnung.core import age_basis, logistic_fit, auc, interval

BASE=['age','sex_numeric','heart_rate','qrs_duration_ms','qtc_fridericia_ms']
EXTRA=['classical_cornell_product','classical_sokolow_lyon','magnitude_area_qrs','spatial_low_voltage_qrs_peak_magnitude','classical_right_axis_deviation','spatial_qrs_t_angle_deg']
MARKER='qrs_t_normalized_discrete_frechet'
RV='qrs_t_normalized_dtw_distance'

def b4(f,comparator='qrs_duration_ms'):
    axis=np.deg2rad(f.classical_right_axis_deviation.to_numpy(float))
    parts=[age_basis(f.age),f[['sex_numeric','heart_rate','qrs_duration_ms','qtc_fridericia_ms']].to_numpy(float),
        f[EXTRA[:4]+['spatial_qrs_t_angle_deg']].to_numpy(float),np.sin(axis),np.cos(axis)]
    if comparator!='qrs_duration_ms':parts.append(f[comparator].to_numpy(float))
    return np.column_stack(parts)

class Fit:
    def fit(self,x,y,w=None):
        x=np.asarray(x,float);y=np.asarray(y,float);w=np.ones(len(y)) if w is None else np.asarray(w,float)
        if not np.isfinite(x).all() or set(np.unique(y))!={0.,1.} or min(np.sum(w[y==v]) for v in (0,1))<=0:raise ValueError('Fit-Eingang/Klassen')
        self.mean=np.average(x,0,weights=w);self.sd=np.sqrt(np.average((x-self.mean)**2,0,weights=w));self.keep=self.sd>1e-12
        self.sd[~self.keep]=1
        z=self.design(x)
        if np.linalg.matrix_rank(z)<z.shape[1]:raise ValueError('Rangdefizit')
        self.beta=logistic_fit(z,y,w);return self
    def design(self,x):return np.column_stack([np.ones(len(x)),((np.asarray(x)-self.mean)/self.sd)[:,self.keep]])
    def score(self,x):return self.design(x)@self.beta
    def prob(self,x):return expit(self.score(x))
    def raw_marker_beta(self):
        if not self.keep[-1]:raise ValueError('Konstanter Marker')
        return float(self.beta[-1]/self.sd[-1])
    def dump(self):return {'mean':self.mean.tolist(),'sd':self.sd.tolist(),'keep':self.keep.tolist(),'beta':self.beta.tolist(),'ridge':1e-6,'iterations_max':60,'score_tolerance':1e-7}

def draw(y,rng):
    w=np.zeros(len(y))
    for v in (0,1):
        ids=np.flatnonzero(y==v);w+=np.bincount(rng.choice(ids,len(ids),replace=True),minlength=len(y))
    return w

def compare(f,y,marker,weights=None,replicates=1000,seed=1212026,bootstrap_weights=None,comparator='qrs_duration_ms'):
    # Eine Person/ein EKG; somit Zeilenbootstrap = Patientenbootstrap.
    assert not f.subject_id.duplicated().any()
    assert set(f.split)<= {'train','validation'}
    x=b4(f,comparator);m=np.asarray(marker,float);y=np.asarray(y,float);tr=f.split.eq('train').to_numpy();va=~tr
    count={s:{'n':int(mask.sum()),'positive':int(y[mask].sum()),'negative':int((y[mask]==0).sum())} for s,mask in [('train',tr),('validation',va)]}
    if any(min(c['positive'],c['negative'])<20 for c in count.values()):return {'status':'nicht_schaetzbar','reason':'weniger als 20 je Klasse/Fold','counts':count}
    wt=np.ones(len(f)) if weights is None else np.asarray(weights,float)
    full=np.column_stack([x,m]);sd=np.std(m[tr])
    if not np.isfinite(sd) or sd<=1e-12:return {'status':'nicht_schaetzbar','reason':'konstanter/ungültiger train-Marker','counts':count}
    def calc(w):
        base=Fit().fit(x[tr],y[tr],w[tr]);extended=Fit().fit(full[tr],y[tr],w[tr])
        ab=auc(y[va],base.score(x[va]),w[va]);ae=auc(y[va],extended.score(full[va]),w[va])
        direction=np.sign(extended.raw_marker_beta())
        raw=auc(y[va],direction*m[va],w[va])
        return np.array([ab,ae,ae-ab,extended.raw_marker_beta()*sd,raw]),base,extended
    try:point,base,extended=calc(wt)
    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
        return {'status':'nicht_schaetzbar','reason':str(exc),'counts':count}
    rng=np.random.default_rng(seed);boot=[];failed={}
    for i in range(replicates):
        if i%100==0:
            from werkzeuge.nachrechnung.resources import available
            print(__import__('json').dumps({'phase':'paired_bootstrap','replicate':i,'n':len(f),'memory':available()}),flush=True)
        w=np.zeros(len(f));w[tr]=draw(y[tr],rng);w[va]=draw(y[va],rng)
        try:
            finalw=w*wt if bootstrap_weights is None else bootstrap_weights(rng)
            boot.append(calc(finalw)[0])
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as e:failed[str(e)]=failed.get(str(e),0)+1
    z=np.array(boot)
    result={'status':'completed' if len(z)>=.95*replicates else 'bootstrap_unzureichend','counts':count,'replicates':replicates,'successful':len(z),'failures':failed,
      'baseline_auc':float(point[0]),'augmented_auc':float(point[1]),'delta_auc':float(point[2]),'beta_per_train_sd':float(point[3]),'or_per_train_sd':float(np.exp(point[3])),
      'raw_auc_train_direction':float(point[4]),'train_sd':float(sd),'direction':int(np.sign(point[3])),'comparator':comparator,
      'baseline_fit':base.dump(),'extended_fit':extended.dump(),'ci_target':'Train-/validation-Patienten getrennt klassenstratifiziert, vollständige Neufits; IPW zunächst bedingt auf eingefrorenes Gewichtsmodell'}
    if len(z):
        result.update({k+'_ci95':interval(z[:,j]) for j,k in enumerate(('baseline_auc','augmented_auc','delta_auc','beta_per_train_sd','raw_auc_train_direction'))})
        result['or_per_train_sd_ci95']=np.exp(result['beta_per_train_sd_ci95']).tolist()
    return result
