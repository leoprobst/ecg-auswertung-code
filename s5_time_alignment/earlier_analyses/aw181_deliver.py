"""AW181: neue Fassungen ausschließlich in zugewiesenen Zielen; Aggregate und Belegspur."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
from repo_config import check_reference, reference
import sys,json,csv,re,hashlib,subprocess
import psutil
assert psutil.virtual_memory().available>=2*2**30, '2 GiB Reserve unterschritten'
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];OUT=Path(str(data_path("s5.aw181")));PUB=ROOT/'publikationen/PUB-08_zeitausrichtung'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,s):Path(p).write_text(s.rstrip()+'\n',encoding='utf-8')
def fmt(v):return f'{v:,}' if isinstance(v,int) else f'{v:.1f}' if isinstance(v,float) else str(v)
def percent(a,b):return f'{100*a/b:.1f}%'
def table(headers,rows,caption):
 return '\n\nTable: '+caption+'\n\n| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in rows)+'\n\n'
def section(text,start,end,replacement):
 a=text.index(start);b=text.index(end,a);return text[:a]+replacement.rstrip()+'\n\n'+text[b:]
def replace(text,old,new):
 assert old in text,old[:80]
 return text.replace(old,new)
off=load(OUT/'offset_results.json');match=load(OUT/'matching_results.json');af=load(OUT/'af_results.json');boundary=load(OUT/'admission_boundary.json')
historical=load(str(data_path("s5.uhrzeit_aw110_icu_results_json")))
def afrow(fold,stage,horizon=365):return next(r for r in af if r['split']==fold and r['stage']==stage and r['horizon_days']==horizon)
def tie(fold,rule):return next(r for r in off['ties'] if r['fold']==fold and r['rule']==rule)
labels={'train':'Development','validation':'Internal assessment'}

# Complete aggregate data behind Figure 1A, no patient or recording identifiers.
csvpath=PUB/'supplementary_data_S5_S1_population_curve.csv'
with csvpath.open('w',encoding='utf-8',newline='') as f:
 w=csv.writer(f);w.writerow(['partition','offset_minutes','median_person_median_absolute_HR_difference_beats_per_minute','people_with_complete_curve'])
 for fold in ['train','validation']:
  curve=historical['folds'][fold]['population_curve'];assert len(curve)==289
  for delta,value in zip(range(-720,721,5),curve):w.writerow([fold,delta,f'{value:.6f}',historical['folds'][fold]['complete_people']])

# All fresh numerical cells have explicit source, key, script, denominator and proof level.
newclaims=[]
sources=[('offset',off,OUT/'offset_results.json'),('matching',match,OUT/'matching_results.json'),('AF',af,OUT/'af_results.json'),('boundary',boundary,OUT/'admission_boundary.json')]
def walk(value,key,source,fold='',denom=''):
 if isinstance(value,dict):
  fold=value.get('fold',value.get('split',fold));denom=str(value.get('n',value.get('jointly_identifiable',denom)))
  for k,v in value.items():
   if k=='selection_steps':continue
   walk(v,key+'.'+k,source,fold,denom)
 elif isinstance(value,list):
  for i,v in enumerate(value):walk(v,key+f'[{i}]',source,fold,denom)
 elif isinstance(value,(int,float)) and not isinstance(value,bool):
  newclaims.append({'aussage':key,'wert':str(value),'originaldatei':str(source),'filter_zeilenschluessel':key,'partition':fold,'nenner':denom,'einheit':'siehe benannter Schlüssel; min, Personen, Kanten oder Anteil','skript':'werkzeuge/s5/aw181_compute.py','originaldatei_sha256':sha(source),'pruefniveau':'neue Entwicklungsrechnung; historische Kernzahlen reproduziert; keine unabhängige Uhrwahrheit','manuskriptstelle':'Supplement S1/S2/S3 bzw. Main Results/Discussion'})
for name,data,source in sources:walk(data,name,source)
with (ROOT/'nachrechnung/s5_ergaenzung.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(newclaims[0]));w.writeheader();w.writerows(newclaims)

# Main text: retain author declarations and AI statement exactly from authorized original.
m=(PUB/'MANUSKRIPT_PREPRINT_S5_v1.md').read_text(encoding='utf-8-sig').replace('Preprint, version 1. This manuscript has not been peer reviewed.\n\n','')
original_ai=m.split('## Use of generative AI\n\n')[1].split('\n\n##')[0]
m=replace(m,'# Temporal support, controlled-pair feasibility and observation processes in MIMIC-IV ECG-EHR studies','# Heart-rate-based clock alignment in intensive-care ECGs, serial-ECG matching and loss of observation in MIMIC-IV')
abstract='''## Abstract

**Objective:** To audit the timing, matching and observation requirements of serial electrocardiogram (ECG) studies using routine clinical records.

**Approach:** We used two person-disjoint, previously inspected development partitions of MIMIC-IV-ECG linked to MIMIC-IV. Intensive-care heart-rate alignment was assessed with fixed identifiability criteria, injected shifts and within-person permutations. Laboratory matching and discharge-landmark observation censuses were examined separately.

**Main results:** In development and internal assessment, respectively, offsets were identifiable for 392 of 17,067 (2.3%) and 145 of 5,846 (2.5%) people. Among these, 67.1% (95% confidence interval 62.2-71.6%) and 70.3% (62.4-77.3%) lay within ±60 min. Injected shifts were recovered conditionally, but permutation also produced identifiable minima. Potassium matching at ±2 h yielded upper bounds of 41 and fewer than 10 sets; at ±6 h, loosening baseline-potassium calipers increased 352 and 69 to 562 and 136. Before marker-completeness selection, 23,151 and 7,945 people had positive observed post-discharge time; at 365 days, contact cessation occurred in 9,769 and 3,356 people without earlier documented atrial fibrillation. A further 5,119 and 1,800 compatible landmarks had no positive observed time.

**Significance:** Physiological alignment does not calibrate individual clocks. Matching counts depend on chosen calipers; cessation of visible contact does not certify event freedom.

Keywords: electrocardiography; electronic health records; time alignment; MIMIC-IV; repeated measurements; data quality'''
m=section(m,'## Abstract','### Evidence status',abstract)
evidence='''### Evidence status

All results use person-disjoint, previously inspected development partitions; no independent held-out confirmation is reported. The atrial-fibrillation (AF) selector was redesigned after inspection of earlier results, including historically opened test data elsewhere in the larger project; no test data were accessed for this note. Heart-rate compatibility estimates are not calibrated clock errors, and matching bounds are not completed treatment-effect samples. Contact cessation does not certify event freedom.'''
m=section(m,'### Evidence status','## 1. Introduction',evidence)
m=replace(m,'MIMIC-IV-ECG provides diagnostic ECGs','MIMIC-IV-ECG provides diagnostic electrocardiograms (ECGs)')
m=replace(m,'Goodwin *et al* (2023) matched heartbeat intervals across clinical waveform systems and accounted for clock drift.','Goodwin *et al* (2023) describe heartbeat-based waveform synchronization and clock drift in their abstract. Fruet *et al* (2026) describe timestamp mapping against a master clock.')
m=replace(m,'We examined a narrower database question: how much individual temporal support can be recovered from routine heart-rate documentation, and how many controlled serial comparisons remain after observation and matching requirements? This note reports technical feasibility and non-identifiability, without estimating treatment effects or diagnostic performance.','We ask one question: what timing, matching and observation information should be reported before interpreting serial ECG-EHR comparisons? Electrolyte-linked recordings are an example because nominal laboratory proximity alone does not establish the order of physiological changes. Comparing potassium-related waveform changes or calcium-related QT changes also requires comparable control recordings. We quantify these requirements without estimating treatment effects or diagnostic performance and provide a reusable reporting table.')
data='''### 2.1. Data and scope

MIMIC-IV-ECG version 1.0 identifies MIMIC-IV version 2.2 as its parent resource (Gow *et al* 2023, Johnson *et al* 2023b). We linked ECG records to MIMIC-IV version 3.1 hospital and intensive-care unit (ICU) modules by the de-identified person key. The ECG documentation states that its shifted times align with clinical-module times; the version 3.1 documentation describes the same person-specific shift across modules (Johnson *et al* 2024). This is documented module consistency, not a new rowwise comparison of versions or a correction of device-clock error. The source ECG resource covers routine care at Beth Israel Deaconess Medical Center, Boston, during 2008-2019; sample-specific calendar distributions were not calculated.

LoopLens (version 10.1; frozen release of 7 October 2026), an in-house Python pipeline for ECG delineation, vectorcardiogram (VCG) reconstruction and marker calculation, supplied stored measurement and context data. Development and internal assessment are two previously inspected, person-disjoint development partitions. The temporal criteria were fixed before calculation; two implementation corrections followed diagnostic runs. Matching-window choices and the AF selector were iterative. The new sensitivity recipes were fixed before their calculations but after the earlier results had been inspected. This work was not prospectively registered.'''
m=section(m,'### 2.1. Data and scope','### 2.2. Heart-rate alignment audit',data)
m=replace(m,'five beats per minute','5 beats per minute');m=replace(m,'one beat per minute','1 beat per minute')
m=replace(m,'The population curve was the median of person-specific curves.','The population curve was the median of person-specific curves (Figure 1A).')
m=replace(m,'The specified criterion required at least 90% of identifiable offsets to lie within ±60 min.','Exact argmin ties preferred the smallest absolute offset, then the smaller signed offset. The specified criteria required a median offset within ±15 min and at least 90% of identifiable offsets within ±60 min.')
m=replace(m,'People without an identifiable offset remained in the availability denominator.','Intervals reflect between-person sampling conditional on the retained estimates; they do not quantify individual clock-error uncertainty or re-estimate curves. People without an identifiable offset remained in the availability denominator.')
m=replace(m,'### 2.3. Pair and matching feasibility','Known shifts of 0, ±30, ±60, ±120 and ±240 min were added to ECG times, retaining the chart series and stay boundaries. Recovery was measured relative to each person’s unshifted estimate, not an unknown true clock. We used 20 within-person heart-rate reassignments (seed 181001) as a permutation diagnostic and compared exact-tie alternatives and near-optimal-zone classifications (Supplement S1).\n\n### 2.3. Pair and matching feasibility')
m=replace(m,'Calipers covered age, initial heart rate, initial potassium and ECG interval.','Calipers covered age, initial heart rate, initial potassium and ECG interval (Table 1). Their exploratory sensitivities are reported in Supplement S2.')
m=replace(m,'The calcium and heart-rate feasibility study','The calcium and heart-rate feasibility study for the QT interval (ventricular depolarization to repolarization)')
m=m.replace('calcium-related QT changes','calcium-related QT-interval changes (ventricular depolarization to repolarization)')
m=m.replace('The calcium and heart-rate feasibility study for the QT interval (ventricular depolarization to repolarization)','The calcium and heart-rate feasibility study')
m=m.replace('Matching-window choices and the AF selector were iterative.','Matching-window choices and the atrial-fibrillation (AF) selector were iterative.')
m=replace(m,'additional QRS, quality and device-label screens','additional QRS-complex duration (ventricular depolarization), quality and device-label screens')
m=replace(m,'The troponin design required supported ECG-to-clinical temporal alignment before ordering ECGs around a laboratory reporting transition. The implementation marked this requirement as unmet throughout the development resource. Counts after this requirement describe the feasibility of that design, not the prevalence of troponin changes.','The troponin reporting-transition design was excluded by rule: its temporal-support flag was false throughout the implementation. It was not an empirical zero-pair finding and is omitted from Table 1; no physiological onset or assay-transition effect is analyzed. Minimum counts of 100 potassium sets per partition and 50 calcium/QT sets per layer and partition were internal operational thresholds, without a power or precision derivation.')
intro_result='''The population heart-rate error minimum was at 0 min in both partitions (Figure 1A). RR-eligible intensive-care ECGs were 42,235 of 477,363 development recordings (8.8%) and 14,233 of 162,841 internal-assessment recordings (8.7%). No emergency-department ECG supported complete ±12 h coverage under this recipe. Complete curves were available for 6,089 of 17,067 and 2,062 of 5,846 people, of whom 1,482 and 497 had at least three complete ECGs.'''
m=replace(m,'The population heart-rate error minimum was at 0 min in both partitions. Complete curves were available for 6,089 of 17,067 and 2,062 of 5,846 people, of whom 1,482 and 497 had at least three complete ECGs.',intro_result)
m=replace(m,'Individual offsets were identifiable for 392 and 145 people.','Individual offsets were identifiable for 392 and 145 people, 2.3% and 2.5% of nominal ICU-linked people and 0.41% and 0.45% of all people in the two ECG development partitions.')
m=replace(m,'Both partitions failed the specified 90% criterion.','Both partitions failed the specified 90% within-hour criterion, while the median-offset criterion was met (0 min in each).')
m=replace(m,'Among people with an identifiable offset, 46.2% and 47.6% lay within ±15 min and 88.0% and 91.0% within ±360 min; the specified 90% criterion applies to ±60 min only.','Among identifiable people, 75.0% and 80.0% lay within ±120 min and 88.0% and 91.0% within ±360 min. These are compatibility shares, not the proportions of ECG-laboratory pairs retained in ±2 h or ±6 h windows. The ±15 min shares (46.2% and 47.6%) are finer than the typical near-optimal widths: median 35 min (interquartile range 15-55) and 45 min (20-55).')
shiftmin=min(r['within60']/r['jointly_identifiable'] for r in off['shifts'] if r['shift_minutes'])
newresult=f'''\n\nInjected shifts passed the fixed differential-recovery check in both partitions: at least {100*shiftmin:.1f}% of jointly identifiable estimates recovered the change within 60 min, with denominator losses and signed bias (Supplement S1). Permutations nevertheless produced 287-351 and 94-128 identifiable minima, so identifiability does not establish clock truth. Exact-tie midpoints gave ±60 min shares of 63.8% and 68.3%, versus 67.1% and 70.3% originally; entire near-optimal zones lay inside ±60 min for 54.8% and 57.2%.'''
m=replace(m,'Figure 1 summarizes the population curve, the reduction from nominal linkage to identifiable offsets and the window shares.','Figure 1 summarizes the population curve, the reduction from nominal linkage to identifiable offsets and the window shares.'+newresult)
m=replace(m,'The ±60 min proportions were 65.1% in both partitions.','The ±60 min proportions were 65.1% in both partitions (person-cluster 95% intervals 60.3-70.0% and 56.8-73.0%); the stay-level population minima occurred at 5 min.')
m=replace(m,'**Table 1. Feasibility counts in the two development partitions.** Each matched set links an eligible change opportunity with a separate control opportunity. Counts from different rows are not additive.','Table: Table 1. Feasibility counts in the two development partitions.')
m=re.sub(r'^\| Troponin reporting transition.*\n','',m,flags=re.M)
m=replace(m,'≤ marks upper bounds from generous matching. Development and internal assessment are the two person-disjoint partitions of the development resource.','Table note: Each set links change and control opportunities from different people. ≤ marks generous upper bounds. Potassium initial pairs precede quality screening; calcium initial pairs are already quality/QRS/device screened (before those screens, 2,062 and 697 pairs). Rows are not additive; development and internal assessment are person-disjoint development partitions.')
m=replace(m,'The potassium feasibility criterion of at least 100 matched sets per partition could not be met even by the wider-window upper bound in the internal assessment partition.','At the original calipers, the ±6 h development upper bound exceeded the internal 100-set threshold, whereas internal assessment did not. Relaxing baseline potassium from 0.3 to 0.5 and 1.0 mmol/L increased ±6 h bounds to 425 and 562 sets in development and 99 and 136 in internal assessment. The latter meets the numerical threshold only under the relaxed rule; treatment and rhythm eligibility remain unresolved. At ±2 h, even combined relaxed calipers yielded only 134 and 37 sets (Supplement S2).')
afrows=[]
for fold in ['train','validation']:
 for stage in ['at_risk','complete_cases']:
  r=afrow(fold,stage);afrows.append([labels[fold]+(' before completeness' if stage=='at_risk' else ' complete cases'),r['n'],r['events'],r['non_AF_contact_end'],r['non_AF_death_end'],r['non_AF_horizon_end'],r['non_AF_tied_end']])
afrows.extend([['Development, no positive observed time',5119,'-','-','-','-','-'],['Internal assessment, no positive observed time',1800,'-','-','-','-','-']])
aftext='''The AF example is a census of stored discharge landmarks, not an incidence analysis or prognostic model. The first eligible discharge and last eligible reported-sinus ECG with regular waveform metadata 0-7 days before discharge were selected. Known prior/index AF and AF through discharge were excluded; risk time began at discharge. AF meant first later documentation, using diagnosis-linked admission-time proxies or ECG-report times, rather than adjudicated onset. Observation ended at the earliest documented AF, visible hospital/ECG contact end, recorded death or horizon. The historical recipe censored death, although it is a competing event. No continuous rhythm surveillance, fixed disease-free washout or complete out-of-system coverage was established.

Selected landmarks numbered 28,297 and '''+reference("aw181_deliver.landmarks_selected_internal","[n1]")+'''. Compatible references remained for 28,270 and 9,745; 5,119 and 1,800 then lacked positive observed post-discharge time. Risk persons before marker completeness numbered 23,151 and 7,945; complete marker, age, administrative-sex and heart-rate data retained 15,092 and 5,257, removing 34.8% and 33.8%. Selection upstream of these landmarks was not fully reconstructed. Table 2 and Supplement S3 separate the risk-person and complete-case censuses.'''
aftext+=table(['Partition and stage','People','Documented AF','Contact end','Death end','Horizon end','Tied end'],afrows,'Table 2. Observation at 365 days and the preceding loss of positive observed time; not incidence estimates.')
aftext+='''Table note: AF = atrial fibrillation. Contact, death, horizon and tie columns apply to people without earlier documented AF. Each risk/complete-case row partitions its person denominator exactly. The last two rows are pre-risk selection losses, with no end-reason classification. Ties mean identical minimum timestamps among contact, death and horizon; the historical event rule is retained.

Contact end among people without AF was 43.6% and 43.7% before completeness, and 43.5% and 44.0% after completeness. Recorded deaths within 365 days numbered '''+reference("aw181_deliver.deaths_recorded_dev_riskset","[n2]")+''' and '''+reference("aw181_deliver.deaths_recorded_val_riskset","[n3]")+''' before completeness and 997 and 326 after completeness; deaths strictly after contact cessation numbered '''+reference("aw181_deliver.deaths_after_contact_dev_riskset_old","[n4]")+''' and 418, and '''+reference("aw181_deliver.deaths_after_contact_dev_complete_old","[n5]")+''' and 240, respectively. Death dates have day-level precision, so exact intraday ordering is unavailable. Recorded-death totals are not death-end counts and their difference does not identify post-contact deaths.

Contact cessation marks loss of visible documentation, not all care or verified event freedom. Documentation of AF is mechanically coupled to contact. Exact finite counts have no inferential confidence intervals. Earlier selector changes followed inspected results and do not provide replication or held-out performance evidence.'''
aftext=aftext.replace(reference("aw181_deliver.deaths_after_contact_dev_riskset_old","[n4]"),reference("aw181_deliver.deaths_after_contact_dev_riskset_new","[n6]")).replace(reference("aw181_deliver.deaths_after_contact_dev_complete_old","[n5]"),reference("aw181_deliver.deaths_after_contact_dev_complete_new","[n7]"))
check_reference("aw181_deliver.death_after_contact_train_at_risk",afrow('train','at_risk')['death_after_contact_before_horizon'])
check_reference("aw181_deliver.death_after_contact_train_complete_cases",afrow('train','complete_cases')['death_after_contact_before_horizon'])
m=section(m,'The AF example is a census','## 4. Discussion',''+aftext)
m=replace(m,'A large database can supply many initial pairs while offering few comparable controls.','At ±6 h, the implemented calipers retained 1,194 person edges from 703 change and 3,104 control people in development. Baseline-potassium relaxation had a larger effect than relaxing heart rate, age or gap class separately (Supplement S2). Sparse matching is conditional on these choices, not intrinsic infeasibility of the database.')
m=replace(m,'The 5 min grid limits resolution.','The 5 min grid is finer than effective resolution: nearest charting allows 30 min distance, and median near-optimal widths were 35 and 45 min. Minute-zero timestamps comprised 87.2% of median-reduced chart observations in each partition; a minute-zero timestamp alone does not establish rounding.')
m=replace(m,'Informative presence and observation are established concerns in EHR studies','Informative presence and observation are established concerns in routinely collected health data')
m=replace(m,'Our decision sequence describes these limitations and is not a validated correction for informative censoring.','Our reporting sequence describes these limitations and is not a validated correction for informative censoring. Identifiability increased across ECG heart-rate variability tertiles (Supplement S1), consistent with a physiology-dependent compatibility measure. Descriptive unit/device/pseudo-calendar strata provide no hardware attestation. A coarse admission-boundary check found 65,799 of 243,432 and 22,638 of 83,347 admission-linked ECGs outside the assigned admission interval; contextual admission links and out-of-hospital recordings can contribute, so these are not clock-error rates.')
reporttable=table(['Item','Minimum report'],[
 ['Population and units','Report recordings, pairs, people and matched sets separately, with exclusive losses and pre-marker denominators.'],
 ['Time semantics','Distinguish recording, specimen/event, storage and administration times; state version linkage and nominal versus independently supported ordering.'],
 ['Alignment','Give candidate range, coverage, tie rule, individual uncertainty, positive/permutation diagnostics and the population without estimates.'],
 ['Controlled opportunities','Specify control recordings, calipers and sensitivity counts; distinguish bounds/proxies from full clinical eligibility.'],
 ['Observation','Define risk start, documented endpoint, last contact, death and horizon; show no-positive-time losses and completeness selection.'],
 ['Evidence and decisions','Disclose inspected partitions, post-inspection choices, operational thresholds, limitations and independent validation status.']],
 'Table 3. Minimum items to report in serial ECG-EHR studies.')
m=replace(m,'## 5. Conclusion',reporttable+'### Clinical interpretation\n\nThis audit informs study design and reporting. It does not support patient-level clock correction or a change in clinical management.\n\n### What this study does not show\n\nNo calibrated device-clock error, treatment effect, diagnostic performance or AF incidence is established. A favorable population curve or relaxed matching count does not resolve unobserved clinical eligibility.\n\n## 5. Conclusion')
m=replace(m,'Nominal linkage and population heart-rate agreement did not provide sufficient individual temporal support for the audited short-term designs.','Nominal linkage and population heart-rate agreement did not establish individual clock accuracy or general ≤1 h ordering for the audited intensive-care subset. Conditional compatibility shares at ±2 h and ±6 h do not validate laboratory-pair simultaneity; matching feasibility depends on chosen calipers.')
m=replace(m,'An event-free label also requires an explicit observation contract','An event-free label also requires explicit observation requirements')
m=replace(m,'## Acknowledgments','## Acknowledgements');m=replace(m,'## Ethics','## Ethical statement')
m=replace(m,'required CITI training','required Collaborative Institutional Training Initiative (CITI) training')
m=replace(m,'Consistent with published secondary analyses of these databases, no additional ethics approval or informed consent was required for this analysis.','No additional ethics approval or informed consent was required for this analysis.')
m=replace(m,'the aggregate values underlying Tables 1-2, Figure 1 and the stay-level counts are provided in this note and the accompanying Supplement.','aggregate table values, Figure 1B-C values and stay-level counts are provided in this note and the accompanying Supplement; the complete Figure 1A curve is supplied as Supplementary Data S1. An internal claim matrix retains numerical provenance and is not a public data repository.')
m=replace(m,'the agreement-reporting principles of GRRAS (Kottner *et al* 2011)','relevant agreement-reporting principles of the Guidelines for Reporting Reliability and Agreement Studies (GRRAS; Kottner *et al* 2011)')
m=replace(m,'the observational-data principles of STROBE (von Elm *et al* 2007) and RECORD (Benchimol *et al* 2015)','the observational-data principles of Strengthening the Reporting of Observational Studies in Epidemiology (STROBE; von Elm *et al* 2007) and the REporting of studies Conducted using Observational Routinely-collected health Data statement (RECORD; Benchimol *et al* 2015)')
m=replace(m,'Related manuscripts and their current status will be disclosed at submission. No publication status is inferred from an internal draft.','No publication status is inferred from an internal draft.')
m=replace(m,'Users should define the temporal ordering their question requires before choosing a laboratory window. A window describes nominal proximity; it does not certify physiological simultaneity. For slowly changing states, a nominal analysis may be useful if its assumptions and fixed window sensitivities are explicit. Enlarging a window after observing sparse counts can alter both the study population and treatment exposure. It should be reported as an exploratory design choice, without treating the new result as confirmation of the original design.','Laboratory windows describe nominal proximity, not physiological simultaneity. Enlarging a window after inspecting sparse counts can change the population and treatment exposure; it is an exploratory design choice, not confirmation of the original design.')
m=replace(m,'The proposed contribution is therefore an empirical feasibility audit of nominal ECG-to-EHR linkage. General synchronization and timestamp-quality problems are already known. We do not report the discovery of unsynchronized ECG clocks, a validated universal matching window or calibrated individual timestamps.\n\n','')
m=replace(m,'Serial comparisons of separate ECGs also need control pairs from different recordings, with no change of clinical state and a comparable interval. A future controlled analysis should estimate the difference between change-pair and control-pair differences, with uncertainty that accounts for repeated observations within people.','Serial analyses need control pairs from distinct recordings without a state change and with comparable intervals. Compare change-pair with control-pair differences using person-cluster uncertainty.')
m=replace(m,'Gow *et al* 2023); the ECG','Gow *et al* 2023); the ECG') if 'Gow *et al* 2023); the ECG' in m else m
m=replace(m,'Gow B *et al* 2023 MIMIC-IV-ECG','Fruet D, Cimignolo S and Nollo G 2026 Multimodal data synchronization: a high-level software methodology for heterogeneous devices *BMC Biomed. Eng.* **8** 12 (doi: 10.1186/s42490-026-00114-x)\n\nGow B *et al* 2023 MIMIC-IV-ECG')
# Distinguish two Johnson 2023 entries consistently; all paper citations become 2023a.
m=m.replace('Johnson *et al* 2023, 2024','Johnson *et al* 2023a, 2024').replace('Johnson *et al* 2023)','Johnson *et al* 2023a)')
m=replace(m,'Johnson A E W *et al* 2023 MIMIC-IV','Johnson A E W *et al* 2023a MIMIC-IV')
m=replace(m,'Johnson A, Bulgarelli L, Pollard T, Gow B','Johnson A, Bulgarelli L, Pollard T, Horng S, Celi L A and Mark R 2023b MIMIC-IV (version 2.2) *PhysioNet* (doi: 10.13026/6mm1-ek67)\n\nJohnson A, Bulgarelli L, Pollard T, Gow B')
assert m.split('## Use of generative AI\n\n')[1].split('\n\n##')[0]==original_ai
mainpart,refs=m.split('## References\n\n',1)
m=mainpart+'## References\n\n'+'\n\n'.join(sorted(refs.strip().split('\n\n'),key=str.casefold))+'\n'
write(PUB/'MANUSKRIPT_S5_v10.md',m)
pre=m.replace('\n\nthe author','\n\nPreprint, version 1. This manuscript has not been peer reviewed.\n\nthe author',1)
write(PUB/'MANUSKRIPT_PREPRINT_S5_v2.md',pre)

# Supplement: retain checklist content while adding all variants and reducing internal provenance to recipes.
s=(PUB/'SUPPLEMENT_S5_v1.md').read_text(encoding='utf-8-sig')
s=s.replace('Supplement version 1','Supplement version 2').replace('No new patient-level calculation was performed for this document revision. ','')
s=s.replace('Charted ICU heart rate','Charted intensive-care unit (ICU) heart rate',1).replace('beats/min','beats per minute').replace('beat/min','beat per minute')
s=s.replace('The specified agreement criterion was ≥90% of identifiable offsets within ±60 min.','Exact argmin ties prefer the smallest absolute offset, then the smaller signed offset. The specified criteria were median offset within ±15 min (met: 0 min in each partition) and ≥90% of identifiable offsets within ±60 min (unmet).')
s=s.replace('The population curve has no computed confidence band.','These intervals condition on the stored point estimates and do not measure individual clock uncertainty or re-estimate identifiability curves. The population curve has no computed confidence band.')
s=s.replace('The original reports contain the cluster intervals; the manuscript does not infer improvement by comparing percentages across person and stay denominators.','Person-cluster 95% intervals were 7.0-8.6% and 7.3-10.4% for stay identifiability; 6.1-12.1% and 2.3-10.7% for half-agreement evidence; and 60.3-70.0% and 56.8-73.0% for conditional ±60 min shares. The stricter split requiring at least three ECGs per half yielded 18 and fewer than 10 estimable stays, with fewer than 10 agreeing in each partition, among 95 and 37 identified stays with at least six ECGs. Person and stay denominators are not interchangeable.')
addition='''### S1.4. Exploratory operating-characteristic diagnostics

Recipes were fixed after inspection of earlier results and before these computations. ECG timestamps were shifted by 0, ±30, ±60, ±120 and ±240 min; chart times, nominal stay boundaries and the original complete-coverage/identifiability rules were retained. The input population had at least three complete baseline ECG curves (1,482 and 497 people). Expected shifted estimates equal baseline estimate minus the injected shift. Bias is the mean signed difference from that expected estimate; recovery shares and zone inclusion are conditional on identifiability at both settings. A near-optimal zone is not a confidence interval. Zero shift is a reproduction check, not known correct clock truth. Positive-control success was defined as median absolute differential error ≤15 min and ≥90% within 60 min in each shift/partition; every variant met that operational rule, which does not calibrate absolute clock error.
'''
rows=[]
for r in off['shifts']:
 rows.append([labels[r['fold']],r['shift_minutes'],r['baseline_at_least3'],r['shifted_at_least3'],r['shifted_identifiable'],r['jointly_identifiable'],r['bias_minutes'],r['median_absolute_error_minutes'],percent(r['within15'],r['jointly_identifiable']),percent(r['within60'],r['jointly_identifiable']),percent(r['zone_covers_expected'],r['jointly_identifiable'])])
addition+=table(['Partition','Injected min','Baseline ≥3','Shifted ≥3','Shifted identifiable','Both identifiable','Bias min','Median absolute error min','Within 15 min','Within 60 min','Zone contains expected'],rows,'Table S1.4. Recovery of injected shifts relative to unshifted fitted offsets.')
addition+='''Twenty permutations (seed 181001) randomly reassigned ECG-derived heart rates to ECG times within each person. Chart times and physiological series remained unchanged. This breaks the within-person ECG-time/heart-rate association while retaining person-level heart-rate distributions. Physiological stability and autocorrelation limit the null: these are diagnostics, not calibrated false-positive probabilities or hypothesis tests.
'''
nullrows=[]
for fold in ['train','validation']:
 rr=[r for r in off['permutations'] if r['fold']==fold];frac=[r['within60']/r['identifiable'] for r in rr]
 nullrows.append([labels[fold],rr[0]['n'],20,f"{min(r['identifiable'] for r in rr)}-{max(r['identifiable'] for r in rr)}",f"{100*min(frac):.1f}-{100*max(frac):.1f}%"])
addition+=table(['Partition','Baseline ≥3 people','Permutations','Identifiable count range','Within ±60 among identifiable range'],nullrows,'Table S1.5. Within-person permutation diagnostic.')
addition+='''### S1.5. Ties, uncertainty zones and effective resolution

Exact argmin ties occurred in 324 of 392 and 121 of 145 identifiable people. The tie-midpoint rule uses the midpoint of the first and last exact argmin candidate, even if that midpoint is not itself a minimizer. Near-optimal-zone endpoints/midpoint are separate sensitivities. Excluding exact ties changes the denominator. Whole-zone and overlap rules classify the original near-optimal zone; neither is a probabilistic confidence statement.
'''
tierows=[]
for r in off['ties']:
 tierows.append([labels[r['fold']],r['rule'].replace('_',' '),r['n'],fmt(r.get('median_offset','-')),str(r.get('within120','-')),str(r.get('within360','-')),str(r['within60']),percent(r['within60'],r['n'])])
addition+=table(['Partition','Rule','People','Median min','Within ±120 count','Within ±360 count','Within ±60 count','Within ±60 share'],tierows,'Table S1.6. Argmin and near-optimal-zone sensitivities.')
addition+='''Near-optimal widths were median 35 min (interquartile range 15-55) and 45 min (20-55). The ±15 min share is finer than this typical region. Median-reduced chart observations numbered 2,941,325 and 1,001,797; 2,564,374 (87.2%) and 873,796 (87.2%) had minute zero, and all had second zero. These counts concern chart observations of the nominal ICU cohort, not raw documentation acts; minute zero alone does not establish timestamp rounding. The ECG heart-rate standard deviation was calculated across baseline complete recordings, and tertile cutoffs were partition-specific. Strata are descriptive and conditioned on at least three complete curves; no rhythm/pacing classification was inferred from RR values.
'''
addition+=table(['Partition','Stratifier','Group','People','Identifiable','Share'],[[labels[r['fold']],'ECG heart-rate SD' if r['variable']=='hf_sd' else 'Complete ECG count',('lower/middle/upper tertile'.split('/')[r['level']] if r['variable']=='hf_sd' else ['3','4-5','≥6'][r['level']]),r['n'],r['identifiable'],percent(r['identifiable'],r['n'])] for r in off['strata']],'Table S1.7. Identifiability by physiology and repeated-recording availability.')
addition+='Tertile boundaries for ECG heart-rate SD were 9.42 and 17.99 beats per minute in development and 9.26 and 17.58 in internal assessment.\n\n'
addition+='''### S1.6. Aggregate source data and descriptive heterogeneity

Supplementary Data S1 contains all 289 offset/error pairs per partition behind Figure 1A, with the complete-curve person denominators. Selected points show the width of the population minimum; no curve confidence band was computed.
'''
points=[]
for fold in ['train','validation']:
 for delta in [-720,-360,-240,-120,-60,-30,-15,0,15,30,60,120,240,360,720]:
  points.append([labels[fold],delta,float(historical['folds'][fold]['population_curve'][(delta+720)//5])])
addition+=table(['Partition','Offset min','Population absolute heart-rate error, beats per minute'],points,'Table S1.8. Selected points of the person-level population curve.')
stratarows=[]
for dimension in ['year','unit','cart']:
 for fold in ['train','validation']:
  for group,rec in historical['strata'][fold+'_'+dimension].items():
   # Descriptive device labels, not person identifiers. Small groups are already collapsed by historical rule.
   if dimension=='cart':group='Device label '+group if group!='unter_30' else 'Labels with <30 complete-curve people'
   else:group=group.replace('unter_30','Groups with <30 complete-curve people')
   stratarows.append([labels[fold],{'year':'Anchor-year group','unit':'First ICU unit','cart':'Device label'}[dimension],group,rec['complete_people'],rec['identifiable_people'],percent(rec['within']['60']['count'],rec['identifiable_people']) if rec['identifiable_people'] else 'not estimable'])
addition+=table(['Partition','Dimension','Group','Complete-curve people','Identifiable','Conditional ±60 share'],stratarows,'Table S1.9. Descriptive anchor-year, first-unit and device-label strata; small groups pooled.')
addition+='''Strata use the most frequent label across each person’s complete ECG curves; each person appears once per dimension. Labels with fewer than 30 complete-curve people are pooled, but identifiable subsets may still be small. These counts describe complete-curve availability, not all ICU people. Device labels do not establish manufacturer or physical recorder identity; first ICU unit is not necessarily the location at ECG acquisition. Anchor-year groups and shifted dates do not recover a real device calendar. Cart-by-shifted-pseudo-month pooling yielded conditional ±60 shares of 64.6% and 62.0%; it cannot validate true device-month stability.

The coarse admission-boundary check compared nominal ECG time with the stored assigned hospital admission. Among 243,432 and 83,347 linked ECGs, 65,799 and 22,638 were outside that interval; none lacked admission boundaries. This is a context-link check, not a clock-error rate: contextual admission assignment and outpatient recordings can contribute. The ICU-eligible audit itself had zero outside-stay timestamps by construction (42,235 and 14,233 ECGs), so it is not an independent validation of stay timing.

'''
s=replace(s,'## S2. Serial-pair feasibility',addition+'## S2. Serial-pair feasibility')
msens='''### S2.5. Exploratory caliper sensitivities

The same marker-free supported opportunities were rematched using the original person-edge definition. Each sensitivity changes one caliper unless labeled combined. All retain exact partition, potassium item, administrative sex, laboratory timing side, distinct persons and the 0.8-1.25 ECG-interval ratio. The combined variant uses potassium ≤1.0 mmol/L, heart rate ≤15 beats per minute, age ≤15 years and no exact gap class. Stricter variants are also shown. Episode combinations and distinct person edges are different units. Independent maximum flow agreed with one-to-one bipartite matching. Counts remain upper bounds with unresolved treatment/rhythm and mutually exclusive first-opportunity eligibility.
'''
msens+=table(['Window h','Partition','Variant','K caliper mmol/L','HR caliper beats per minute','Age years','Exact gap class','Episode combinations','Person edges','One-control sets / links'],[[r['window_h'],labels[r['fold']],r['variant'].replace('_',' '),r['K_caliper'],r['HR_caliper'],r['age_caliper'],'yes' if r['exact_gap_class'] else 'no',r['episode_combinations'],r['person_edges'],r['max_links']] for r in match],'Table S2.4. Matching sensitivity counts at fixed nominal laboratory windows.')
msens+='''For the two-controls variant, each change-person node has capacity two and each control-person node capacity one: the final column is links, not sets of two fully matched controls or independent change people. Baseline one-to-one sets remain 41/<10 and 352/69; two-control link counts are 54/<10 and 554/93. Persons can occur as candidates in both graph arms in these generous bounds; no fully disjoint clinical cohort is asserted. Baseline potassium is the strongest of the individually varied bottlenecks at ±6 h: loosening it to 1.0 mmol/L permits 562/136 one-control sets, compared with 410/85 for heart rate, 398/84 for age and 368/75 without exact gap class. Caliper relaxation trades comparability for count and is not a new validated effect design. Internal 100/50 thresholds were not derived from power or precision; development alone met the original potassium threshold at ±6 h.

'''
s=replace(s,'## S3. AF observation census',msens+'## S3. AF observation census')
s=s.replace('The finite-positive-duration check removed no further cases after the positive-time stage.','The finite-positive-duration check removed no further cases after the positive-time stage.')
s=s.replace('This supplement reports aggregates from that census and does not reread its person-level cache.','The present sensitivity reproduced complete-case counts and added the risk-person stage before marker completeness from the same development-only cache.')
afaddition='''### S3.3. Census before marker completeness and recorded deaths

The historical risk-set and reported-sinus functions were retained; no prognostic fit or incidence model was run. Complete-case counts and all original end-reason cells were reproduced against the prior census before adding the risk-person layer. The 365-day recorded-death totals 997 and 326 belong to development and internal assessment, respectively. The end-reason counts are disjoint, while recorded-death columns may overlap AF or an earlier contact end and must not be added to end-reason columns.
'''
afaddition+=table(['Partition','Stage','Horizon days','People','AF','Contact','Death end','Horizon end','Tie','Any recorded death','Death after contact','Contact / no AF'],[[labels[r['split']],r['stage'].replace('_',' '),r['horizon_days'],r['n'],r['events'],r['non_AF_contact_end'],r['non_AF_death_end'],r['non_AF_horizon_end'],r['non_AF_tied_end'],r['death_recorded_before_horizon'],r['death_after_contact_before_horizon'],f"{100*r['contact_fraction_no_AF']:.1f}%"] for r in af],'Table S3.3. All observation-end censuses before and after completeness selection.')
afaddition+='''Death after contact requires recorded death strictly later than the last visible contact and within the horizon. Among complete cases at 365 days, this was '''+reference("aw181_deliver.deaths_after_contact_dev_complete_old","[n5]")+''' of 997 deaths (74.2%) and 240 of 326 (73.6%); subtracting death ends from all recorded deaths would instead give '''+reference("aw181_deliver.deaths_difference_dev_complete","[n8]")+''' and 254 and is not the same quantity. Death dates may have only day-level precision, limiting same-day temporal ordering. These calculations do not adjudicate AF occurrence after contact.

As a deliberately extreme arithmetic illustration, treating all 365-day contact-ended complete cases as additionally having AF would change the observed-documentation fractions from 411/15,092 to (411+'''+reference("aw181_deliver.extreme_added_dev","[n9]")+''')/15,092 (2.7-45.1%) and from 141/5,257 to (141+'''+reference("aw181_deliver.extreme_added_internal","[n10]")+''')/5,257 (2.7-45.5%). These are not incidence bounds: unobserved AF can also occur in horizon-completed people, ascertainment is coupled to contact, and no continuous event-free observation was established. Before marker completeness the analogous ranges are 3.3-45.5% and 3.4-45.7%.

'''
afaddition=afaddition.replace(reference("aw181_deliver.deaths_after_contact_dev_complete_old","[n5]")+' of 997 deaths (74.2%)',reference("aw181_deliver.deaths_after_contact_dev_complete_new","[n7]")+' of 997 deaths (75.4%)')
s=replace(s,'## S4. Administrative sex',afaddition+'## S4. Administrative sex')
s=s.replace('in this revision','in this audit').replace('this document revision','this audit')
s=s.replace('STROBE (von Elm et al 2007), RECORD (Benchimol et al 2015), GRRAS (Kottner et al 2011) and SAGER (Heidari et al 2016) are bibliographically verified in the accompanying search protocol. Locations refer to manuscript v9 (M) and this supplement (S).','STROBE (von Elm *et al* 2007), RECORD (Benchimol *et al* 2015), GRRAS (Kottner *et al* 2011) and SAGER (Heidari *et al* 2016) were checked against published records. Locations refer to the accompanying manuscript (M) and this supplement (S).')
s=s.replace('The SAGER publisher correction concerns missing peer-review reports, not revised sex/gender analytic recommendations. ','')
provenance='''### S6.2. Reproducible recipes

Person-level heart-rate alignment uses the offset range and rules in S1, with original bootstrap seed 110; stay pooling retains person clusters and seed 115. Shift and permutation sensitivities use the recipes in S1.4 and seed 181001. Laboratory matching uses S2’s exact categories/calipers and maximum cardinality/flow agreement. AF observation uses S3’s first-eligible-discharge selection, documented-event references and 30/90/365-day end rules, both before and after completeness filtering. These describe computable recipes; upstream raw clinical-event reconstruction remains incomplete.

The aggregate curve is supplied as Supplementary Data S1. Private selectors, matching lists, clinical records and full texts are not distributed. Full LoopLens analysis code remains available on reasonable request under the manuscript’s Code availability statement. The internal claim matrix retains source hashes and row-level aggregate provenance; it is not a public repository deposit. A contemporary hash does not prove every historical generator/input identity.
'''
s=section(s,'### S6.2. Source map','### S6.3. Interpretation safeguards',provenance)
s=s.replace('local 11 October 2026; UTC 10 October','11 October 2026 local time; locked 10 October 2026 at 23:34:49 UTC')
s=s.replace('Fixed identifiability and agreement criteria were not relaxed after failure.','The two diagnostic-run corrections enforced stay boundaries and connected near-optimal regions without changing the criteria. Fixed identifiability and agreement criteria were not relaxed after failure. The specification SHA-256 and full calculation protocol are retained in the internal claim matrix; they are not printed as internal identifiers in this public supplement.')
s=s.replace('AF-v2','revised AF discharge selector').replace('documented-AF v1','earlier documented-AF design').replace('Ca/HF','calcium/heart-rate').replace('calcium/HF','calcium/heart-rate').replace('K/Ca/HF','potassium/calcium/heart rate')
s=s.replace('eMAR presence flag','electronic medication administration record (eMAR) presence flag',1)
s=s.replace('balance/SMD','balance/standardized mean difference (SMD)').replace('PPV/NPV','positive/negative predictive values')
s=s.replace('CITI/DUA','credentialed training/data use agreement').replace('M Ethics','M Ethical statement').replace('event contract','event definition').replace('work-point selection','analysis-choice selection')
s=s.replace('this document revision accessed no test data, footer or result. ','no test data were accessed for this note. ')
s=s.replace('not the primary Table 1 count','not the primary Table 1 count of the manuscript')
s=s.replace('Their verified PMID/DOI pairs and verification depth are recorded in the accompanying search protocol. ','')
s=re.sub(r'(?<!\*)\bet al\b(?!\*)','*et al*',s)
s=s.replace('sealed','fixed').replace('gate(s)','selection criteria')
s=re.sub(r'\bgates\b','selection criteria',s)
s=re.sub(r'\bgate\b','selection criterion',s)
s=re.sub(r'\bF[1-8] ','',s)
# Label all tables lacking captions; retain existing named additions.
lines=s.splitlines();new=[];section_no='1';counts={};flowcount={}
for line in lines:
 ma=re.match(r'## S(\d+)\.',line)
 if ma:section_no=ma.group(1)
 if line.startswith('|') and (not new or not new[-1].startswith('|')):
  # Captions are separated from header by an empty line.
  recent='\n'.join(new[-3:])
  if 'Table: ' not in recent:
   counts[section_no]=counts.get(section_no,0)+1
   new.extend([f'Table: Table S{section_no}.{counts[section_no]}. '+{'1':'Original alignment flows and window counts','2':'Implemented matching rules','3':'Original complete-case census','4':'Age and administrative sex','5':'Reporting checklist','6':'Interpretation safeguards'}[section_no]+'.',''])
 if line=='```text':
  flowcount[section_no]=flowcount.get(section_no,0)+1;new.extend([f'Flow S{section_no}.{flowcount[section_no]}. Development and internal-assessment counts at the stated starting stage.',''])
 new.append(line)
s='\n'.join(new)
write(PUB/'SUPPLEMENT_S5_v2.md',s)

# Original matrix retained, including complete internal run/source history.
with (PUB/'CLAIM_MATRIX_S5_v4.csv').open(encoding='utf-8-sig',newline='') as f:r=csv.DictReader(f);fields=list(r.fieldnames);original=list(r)
for k in ['verwendung_v10','protokoll_sha256_aw181','pruefung_aw181']:
 if k not in fields:fields.append(k)
protocolhash=sha(OUT/'protocol.json')
for r in original:
 r['verwendung_v10']='Historischer Beleg; Text/Table-Zuordnung v10 prüfen; frühere Fassungen unverändert nachvollziehbar'
 r['protokoll_sha256_aw181']=protocolhash;r['pruefung_aw181']='Neue Sensitivitäten separat; keine neue unabhängige Testprüfung'
 if 'troponin' in r.get('aussage','').lower():r['verwendung_v10']='Keine Tabelle-1-Zeile; ausschließlich regelbasierter Ausschluss'
for i,n in enumerate(newclaims,1):
 n=dict(n,claim_id=f'S5-AW181-{i:04d}',lauf='A155-2026-10-04-d6e70e; gebundene Entwicklung',bericht='nachrechnung/S5_ERGAENZUNG_AW181.md',status='explorativ gerechnet; keine Uhrkalibration',verwendung_v10='neue Sensitivität/Aggregatbeleg',protokoll_sha256_aw181=protocolhash,pruefung_aw181='Kernreproduktion/Hash/Spaltenprojektion; technische Neuberechnung',testnutzung_historisch='historische Testexposition berichtet; AW181 ohne Testzugriff')
 original.append(n)
spec_lock=load(str(data_path("s5.uhrzeit_aw110_spec_lock_json")))
original.append({'claim_id':'S5-AW181-SPEC','aussage':'AW110 Spezifikation vor Rechnung gesperrt; zwei diagnostische Implementierungskorrekturen ohne Kriterienänderung','wert':spec_lock['sha256'],'einheit':'SHA256','manuskriptstelle':'Supplement S6.1','originaldatei':str(data_path("s5.uhrzeit_aw110_spec_lock_json")),'originaldatei_sha256':sha(str(data_path("s5.uhrzeit_aw110_spec_lock_json"))),'filter_zeilenschluessel':'sha256; locked_utc='+spec_lock['locked_utc'],'skript':'werkzeuge/uhrzeit_aw110/rechnen.py','partition':'train/validation','protokoll_sha256_aw181':protocolhash,'verwendung_v10':'öffentlicher Text Sperrzeit/Korrekturen; Hash nur interne Matrix','pruefung_aw181':'spec_lock Original gelesen; UHRZEIT_AW110.md §6 Abgleich'})
with (PUB/'CLAIM_MATRIX_S5_v5.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(original)

# Revision register: every review point, distinguish external decisions from completed B-K items.
solutions={
1:'S1.4: alle neun Verschiebungen, Bias/Abdeckung/Nenner; 20 innerhalb-Person-Permutationen. Differenzielle Wiederfindung ausdrücklich keine absolute Uhrenkalibration.',
2:'M2.2/S1.5: originale Argmin-Tie-Regel, exakter Tie-Mittelpunkt/Ausschluss und Near-optimal-Enden/-Mitte sowie ganze Zone/Überlappung berechnet.',
3:'Titel/Abstract/M3.1/Discussion eingegrenzt: ICU, 8,8/8,7% RR-EKGs, 0,41/0,45% aller Personen identifizierbar; ED unter identischer Methode nicht prüfbar.',
4:'Abstract benennt Partitionen, Prozentnenner und 392 von 17.067 / 145 von 5.846.',
5:'Median-Gate 0/0 erfüllt, ±60-Gate verfehlt; ±120 75/80% und ±360 88/91%; Laborpaarfenster nicht als Offsettoleranz validiert; Conclusion ≤1h/ICU.',
6:'Troponin-Zeile aus Table1 entfernt; implementiertes FALSE als regelbasierter Ausschluss, kein Nullbefund. Kein Assay-/physiologischer Übergang behauptet.',
7:'S2.5 elf Varianten je Fold/Fenster: K/HF/Alter strenger/lockerer, Lückenklasse, kombiniert, zwei Kontrollen; Kombinationen/Kanten/Links getrennt. M3.2/Discussion jetzt caliperabhängig.',
8:'M2.3/S2.5 interne 100/50-Schwellen ohne Power-/Präzisionsableitung offengelegt; Entwicklung bei ±6h bestanden, Baseline IA verfehlt, K1.0 IA rechnerisch bestanden.',
9:'Risikopersonen-Zensus 23.151/7.945 vor Vollständigkeit auf drei Horizonten gerechnet; Table2/M3.3/S3.3; 5.119/1.800 ohne positive beobachtete Zeit gezeigt.',
10:'997 train/326 validation gegen R5_census exakt reproduziert; Kontaktende/ohne AF und alle Todeszähler ergänzt. Tatsächlich '+reference("aw181_deliver.deaths_after_contact_dev_complete_old","[n5]")+'/240 nach Kontakt, nicht irreführend '+reference("aw181_deliver.deaths_difference_dev_complete","[n8]")+'/254 Differenz.',
11:'Eine Leitfrage in Introduction; serielle Elektrolyt-EKG-Frage motiviert; wiederverwendbare Table3.',
12:'35(15-55)/45(20-55)min Zonenbreite; ±15-Auflösungsgrenze; Minute-00/Second-00 gezählt, keine behauptete Rundungsursache.',
13:'Vollständige 578 Aggregatzeilen Figure1A als Supplementary DataS1; S1.8 ausgewählte Punkte; Aufenthaltsminimum 5min genannt.',
14:'Vorhandene Personenclusterintervalle und strenge ≥3-EKG-pro-Hälfte-Sensitivität in S1.3; ein KI-Paar in M3.1.',
15:'Zulässige Alternative verwendet: GRRAS ausdrücklich relevante Prinzipien; keine Bland-Altman-/LoA-/Vollchecklistenfreigabe behauptet.',
16:'S1.9 vollständige historische Aggregatstrata nach Ankerjahr/erster Einheit/Gerät mit Pooling <30; Cart×Pseudomonat und Hardwaregrenze genannt.',
17:'S1.7 Terzile ECG-HF-SD sowie vollständige EKG-Zahl; konditionale Nenner; kein erfundener Pacing-/Rhythmusklassifikator.',
18:'Vorhandene Entwicklungstabelle gegen zugeordnete Aufnahmegrenzen geprüft: 65.799/243.432 und 22.638/83.347 außerhalb. Kontextlink/ambulante EKGs möglich, keine Uhrfehlerrate. Orders/ED neues Modul B-A.',
19:'M2.2/S1.2: Bootstrap bedingt auf gespeicherte Schätzer, keine Einzeloffset-KI und keine Kurven-Neuziehung.',
20:'S6.1 Sperrzeit und zwei Korrekturläufe offengelegt; Spezifikations-SHA in interner Matrix/Register statt öffentlichen Hashrest (AW181 fordert Provenienz in Matrix).',
21:'S3.3 extreme arithmetische Szenarien klar keine Inzidenzgrenzen; M3.3 mechanische AF/Kontakt-Kopplung.',
24:'S6.2 neu als Rezept; keine lokalen Pfade, internen Läufe/Dateinamen/IDs; Selektor-/Testhistorie beschreibend statt Versionsjargon.',
25:'Prozessmeldungen gestrichen; einmal Evidenzbox und knappe Entscheidungshistorie; kein for-this-revision im öffentlichen Text.',
26:'Titel auf HR-basiertes Alignment in ICU-EKGs, Matching und Verlust der Beobachtung umgeschrieben.',
27:'Table1-Fußnote trennt Potassium pre-screen/Calcium post-screen und Vorzahlen; Table2-Fußnote definiert exakte End-Ties.',
28:'Evidenzbox beschreibt nach Sicht früherer Ergebnisse überarbeiteten AF-Selektor, historisch testhaltige Vorarbeiten und aktuelle Entwicklung getrennt.'}
rev='# Revision S5 v10 / Preprint-Dateifassung v2\n\n13.10.2026 · MS/PR/DO · Banner bleibt „Preprint, version 1“, da unveröffentlicht. Neue Rechnungen explorativ in train/validation; kein Testzugriff. Umfang AE-191 unverändert.\n\n## Schwachstellenbericht: B-K\n\n| Punkt | Status | Behebung und Stelle |\n|---|---|---|\n'
for i in range(1,31):
 if i in solutions:rev+=f'| SW-{i:02d} | erledigt, fachliche Grenzen bleiben | {solutions[i]} |\n'
rev+='''
## B-A-Punkte, nur Aufgabenliste

- SW-22: Vollständiger upstream AF-Indexfluss aus Pipeline-Sitzung; im Text als Designgrenze offengelegt.
- SW-23: Goodwin-Volltext durch Anwender; die Aussage bleibt explizit auf Abstracttiefe. Zusätzliche Prior-Art-/Beobachtungsliteratur außerhalb der auf Quellenkorrekturen begrenzten AW181-Recherche nicht erweitert.
- SW-18, B-A-Teil: ECG-Orders/zusätzliche ED-Module und externe Uhrreferenz erfordern einen gesonderten Auftrag; bestehende Aufnahmeprüfung erledigt.
- SW-29/SW-20: Journalkommunikation, Spezifikationsbeilage und Einreichungsfreigabe durch Anwender/K. Keine neuen Manuskriptplatzhalter.
- SW-30: Autor bestätigt lokale Ethikregelung/rechtliche Aussage vor Einreichung; Blueprint-Zugang/Funding/COI unverändert, Vergleichsarbeitsformel gestrichen.
- Format 9/19/20/21: Word-/PDF-Bau, Evidenzdatei für den Export, Tabellenbreite/Seitenbildkontrolle und Portal-Wortzählung sind nicht in den erlaubten AW181-Schreibzielen; Markdown-Syntax vorbereitet. Keine Word-Freigabe behauptet.

## Designgrenzen U1-U6

Alle erhalten: keine externe Uhrwahrheit; ICU-/Vollabdeckungsselektion und konstante Offsets; ungeklärte Therapie/Rhythmus/gemeinsame Beats; inspizierte Entwicklung und historisch eröffnete Testvorarbeiten; dokumentationsbasierte AF plus unbekannter upstream Fluss; Pseudokalender statt realer Gerätekalender. Sensitivitäten beheben Berichtslücken, nicht diese Designgrenzen.

## Quellenbericht 1-9

| Punkt | Status | Umsetzung |
|---|---|---|
| Q1 | erledigt | PhysioNet-Originale ECG1.0, MIMIC2.2/3.1; parent2.2, person-key link3.1, dokumentierter gemeinsamer Modulshift; kein eigener Zeilenvergleich der Versionen behauptet; Johnson2023b ergänzt. |
| Q2 | erledigt | accompanying manuscript statt v9; keine nicht mitgelieferte Suchprotokollbehauptung im öffentlichen Supplement. |
| Q3 | erledigt | *et al* kursiv vereinheitlicht. |
| Q4 | erledigt | Nicht benötigten SAGER-Correction-Satz gestrichen, kein unbelegtes Zusatzstatement. |
| Q5 | erledigt als Textkorrektur | Vergleichsarbeiten-Teilsatz gestrichen; rechtlicher Autorpunkt SW30 bleibt B-A. |
| Q6 | erledigt | routinely collected health data statt weiter EHR-studies-Verallgemeinerung. |
| Q7 | erledigt | Fruet2026 gegen esearch/esummary/efetch geprüft, enger Abstractsatz und vollständiger Eintrag ergänzt. |
| Q8 | keine Pflichtänderung | Keine neue Softwarezitierbehauptung; öffentliche Rezepte statt nicht erneut attestierter alter Versionsliste. |
| Q9 | erledigt | S6.2 komplett auf Rezeptebene; interne Provenienz in Matrixv5. |

## Formatbericht 1-21

| Punkt | Status | Umsetzung |
|---|---|---|
| 1 | erledigt | Figure1A vollständige Aggregate als CSV S1, Data availability angepasst. |
| 2 | erledigt | Table1 in Methods, Table2 in Results, Figure1A aufgerufen. |
| 3 | erledigt | Abstract Partitionen explizit, Schrägstrichpaare im Haupttext aufgelöst. |
| 4 | erledigt | Table:-Captions und Table note: für Haupttabellen. |
| 5 | erledigt | Sämtliche Supplementtabellen und Flüsse eindeutig nummeriert. |
| 6 | erledigt | HF durch heart rate; ICU/eMAR/SMD und übrige Kurzformen ausgeschrieben bzw. entfernt; AF-Versionen beschreibend. |
| 7 | erledigt | accompanying manuscript, keine v9-Leserreferenz. |
| 8 | erledigt | Clinical interpretation / What this study does not show und interne Matrix als interne Provenienz genannt. |
| 9 | Textteil erledigt; Export B-A | Evidenzbox auf vier Sätze, Doppelung M2.1 entfernt; keine unzulässige neue Exportdatei. |
| 10 | erledigt | Troponin-Zeile entfällt vollständig. |
| 11 | erledigt | Keine revision-Prozesssprache in Preprint/Supplement. |
| 12 | erledigt | ECG/AF/VCG/QRS/QT sowie Reporting-Leitlinien/CITI im Haupttext definiert. |
| 13 | erledigt | 5/1 beats per minute; Einheitenform vereinheitlicht. |
| 14 | bestehend sachgerecht | Quelle nennt >10 Autoren als et al-Regel; neue Drei-Autoren-Quelle vollständig, bestehende geprüfte Einträge beibehalten. |
| 15 | Provenienz hier nachgeführt | Figure1 unverändert; alte Abbildungsprotokolldatei außerhalb erlaubter Schreibziele nicht verändert. Neue Datenbindung CSV/Matrixv5. |
| 16 | erledigt | Ethical statement/Acknowledgements; Abstractlabels Doppelpunkt. |
| 17 | erledigt im bearbeiteten Text | Öffentliche interne Quellenkarte entfernt, Safeguards ohne F-Kürzel, event definition statt contract. |
| 18 | erledigt | Table1 of the manuscript. |
| 19 | Markdown vorbereitet, Export B-A | Flüsse beschriftet; Word-Seitenkontrolle nicht Gegenstand des erlaubten Schreibumfangs. |
| 20 | Export B-A | Kein manueller Word-Zellverbund; Table1-Syntax erhalten, keine Layoutfreigabe. |
| 21 | geprüft, Portal B-A | Reproduzierbare lokale Wortzählung im Validierungsbericht; Portalregel durch K vor Einreichung. |

## Quellenfehler-Kandidaten

PhysioNet ECG1.0, HTML `ecg_1_0.html`, Reference3 versus Parent Projects: Reference3 nennt MIMIC-IV1.0/2021, Parent Projects2.2. Am Original erneut bestätigt (Abschnitte References/Parent Projects; lokale HTML-Zeilen 663 bzw.897, genaue Zeile im Suchprotokoll). Dokumentationsinkonsistenz, kein Nachweis fehlerhafter Zeitverschiebung. Registerpflege bleibt K.

## Freigabestatus

Rechnungs-/Textauftrag umgesetzt; K-Abnahme, B-A-Entscheide und Export-/Fachredaktion bleiben vor Veröffentlichung erforderlich. Nicht als bereits publiziert/einreichungsfreigegeben ausgeben. KI-Erklärung byteidentisch zum Ursprungsabsatz; keine neuen offenen Manuskriptplatzhalter.
'''
rev=rev.replace(reference("aw181_deliver.deaths_after_contact_dev_complete_old","[n5]")+'/240',reference("aw181_deliver.deaths_after_contact_dev_complete_new","[n7]")+'/240').replace('Zeilen 663 bzw.897','Zeilen 807 bzw. 897')
write(PUB/'REVISION_S5_v10.md',rev)

# Public scan is a diagnostic tool, retain its exact output in private execution trace.
scan=subprocess.run([sys.executable,'-B',str(ROOT/'werkzeuge/oeffentlich_scan.py'),str(PUB/'MANUSKRIPT_PREPRINT_S5_v2.md'),str(PUB/'SUPPLEMENT_S5_v2.md')],capture_output=True,text=True,encoding='utf-8')
write(OUT/'public_scan.txt',scan.stdout);assert scan.returncode==0
validation={'banner':'Preprint, version 1' in pre,'AI_unchanged':m.split('## Use of generative AI\n\n')[1].split('\n\n##')[0]==original_ai,'curve_rows':578,'new_numerical_claims':len(newclaims),'complete_case_reproduced':True,'main_words_sections1_5':len(re.findall(r"\b[\w’'-]+\b",m.split('## 1. Introduction')[1].split('## Acknowledgements')[0])),'abstract_words':len(re.findall(r"\b[\w’'-]+\b",abstract.split('Keywords:')[0])),'test_access':False,'new_placeholders':False,'public_scan':scan.stdout,'files':{str(p):sha(p) for p in [PUB/'MANUSKRIPT_S5_v10.md',PUB/'MANUSKRIPT_PREPRINT_S5_v2.md',PUB/'SUPPLEMENT_S5_v2.md',PUB/'CLAIM_MATRIX_S5_v5.csv',PUB/'REVISION_S5_v10.md',csvpath,ROOT/'nachrechnung/s5_ergaenzung.csv']}}
write(OUT/'delivery_validation.json',json.dumps(validation,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in validation.items() if k not in ['files','public_scan']},ensure_ascii=False))
print(scan.stdout)
