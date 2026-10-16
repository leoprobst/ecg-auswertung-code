"""Gepaarte Clusterinferenz; nur synthetisch in AW-023 ausgeführt.

48 bzw. 64 vorab benannte Spalten, gemeinsame Bootstrapgewichte, alpha=.025.
Studentisierung mit marginaler Bootstrap-SE, zentrierte max-|T|-Verteilung.
Kein Erfolg bei Null-SE oder weniger als 95 % gültigen Ziehungen.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

def paired_auc(y,a,b,weights=None):
    y=np.asarray(y);a=np.asarray(a,float);b=np.asarray(b,float)
    w=np.ones(len(y)) if weights is None else np.asarray(weights,float)
    keep=np.isfinite(a)&np.isfinite(b)&np.isfinite(y)&(w>0)
    if len(np.unique(y[keep]))!=2:return np.nan
    return roc_auc_score(y[keep],a[keep],sample_weight=w[keep])-roc_auc_score(y[keep],b[keep],sample_weight=w[keep])

def simultaneous(point,draws,expected,alpha=.025,min_valid=.95):
    point=np.asarray(point,float);draws=np.asarray(draws,float)
    if len(point)!=expected or draws.ndim!=2 or draws.shape[1]!=expected:raise ValueError('Unvollständige vorab festgelegte Kontrastfamilie')
    valid=np.isfinite(draws).all(axis=1)
    if not np.isfinite(point).all() or valid.mean()<min_valid:raise ValueError('Zu wenige gültige gemeinsame Ziehungen')
    se=draws[valid].std(axis=0,ddof=1)
    if not np.isfinite(se).all() or (se<=0).any():raise ValueError('Null-SE: keine Erfolgsaussage')
    maximum=np.max(np.abs((draws[valid]-point)/se),axis=1)
    critical=float(np.quantile(maximum,1-alpha,method='higher'))
    return {'point':point,'se':se,'lower':point-critical*se,'upper':point+critical*se,'critical':critical,'valid_draws':int(valid.sum()),'alpha':alpha}

def bootstrap(clusters,estimate,expected,strata=None,replicates=10000,seed=20261008):
    """estimate(weights) liefert sämtliche Kontraste; gleiche Personengewichte.

    Strata müssen patientenfest sein, primär Endpunktstatus. Bei longitud.
    Daten ist eine explizite patientenfeste Zuordnung erforderlich.
    """
    clusters=np.asarray(clusters);unique,inverse=np.unique(clusters,return_inverse=True)
    strata=np.zeros(len(clusters),int) if strata is None else np.asarray(strata)
    patient_strata=[]
    for i in range(len(unique)):
        s=np.unique(strata[inverse==i])
        if len(s)!=1:raise ValueError('Stratum nicht patientenfest')
        patient_strata.append(s[0])
    patient_strata=np.asarray(patient_strata)
    groups=[np.flatnonzero(patient_strata==s) for s in np.unique(patient_strata)]
    rng=np.random.default_rng(seed);point=np.asarray(estimate(np.ones(len(clusters))))
    draws=np.empty((replicates,expected))
    for j in range(replicates):
        counts=np.zeros(len(unique))
        for group in groups:
            selected=rng.choice(group,len(group),replace=True)
            counts+=np.bincount(selected,minlength=len(unique))
        draws[j]=estimate(counts[inverse])
    return simultaneous(point,draws,expected)

def icc_sdc(repeats):
    """ICC(A,1), zweifaktoriell absolute Einzelmessungsübereinstimmung.

    Balancierte Wiederholungen; SDwithin enthält reale Physiologie, falls
    diese im Eingangsvertrag nicht konstant gehalten wurde.
    """
    x=np.asarray(repeats,float)
    if x.ndim!=2 or x.shape[0]<3 or x.shape[1]<2 or not np.isfinite(x).all():raise ValueError('Balancierte endliche Wiederholungen erforderlich')
    n,k=x.shape;grand=x.mean();row=x.mean(1);column=x.mean(0)
    msr=k*np.sum((row-grand)**2)/(n-1)
    msc=n*np.sum((column-grand)**2)/(k-1)
    mse=np.sum((x-row[:,None]-column[None,:]+grand)**2)/((n-1)*(k-1))
    denominator=msr+(k-1)*mse+k*(msc-mse)/n
    within=np.sqrt(np.mean(np.var(x,axis=1,ddof=1)))
    return {'icc_a1':(msr-mse)/denominator if denominator>0 else np.nan,'sd_within':within,'sdc':1.96*np.sqrt(2)*within}

def log_sd_ratio(variant,original,train_sd_variant,train_sd_original):
    if min(train_sd_variant,train_sd_original)<=0:raise ValueError('Train-SD fehlt oder Null')
    a=icc_sdc(variant)['sd_within']/train_sd_variant
    b=icc_sdc(original)['sd_within']/train_sd_original
    return float(np.log(a/b)) if min(a,b)>0 else np.nan

def agreement(kors,dower):
    difference=np.asarray(dower,float)-np.asarray(kors,float)
    difference=difference[np.isfinite(difference)]
    if len(difference)<3:raise ValueError('Zu wenige Methodenpaare')
    bias=difference.mean();sd=difference.std(ddof=1)
    return {'bias':bias,'loa_lower':bias-1.96*sd,'loa_upper':bias+1.96*sd}

def invariance(reference,value,fraction=False):
    if not np.isfinite([reference,value]).all():return False
    tolerance=.01 if fraction else 1 if reference==0 else .05*abs(reference)
    return abs(value-reference)<=tolerance

def fold_consistent(train_oof,validation):
    return bool(np.isfinite([train_oof,validation]).all() and train_oof*validation>0)

def decision(clinical,measurement,train_oof,analytical_pass):
    """Spaltenreihenfolge: je Variante Original, Komparator, B2; danach log-SD."""
    if len(clinical['point'])!=48 or len(measurement['point'])!=64:raise ValueError('Familien 48/64 erforderlich')
    out=[]
    for i in range(16):
        j=3*i;consistent=fold_consistent(train_oof[i],clinical['point'][j])
        clinical_gain=consistent and clinical['point'][j]>=.02 and clinical['lower'][j]>0 and clinical['lower'][j+2]>0
        ratio=np.exp(measurement['point'][48+i]);upper=np.exp(measurement['upper'][48+i])
        measurement_gain=ratio<=.8 and upper<1 and measurement['lower'][j]>-.02 and (i<8 or analytical_pass[i])
        out.append({'clinical_gain':bool(clinical_gain),'measurement_gain':bool(measurement_gain),'sd_ratio':float(ratio)})
    return out
