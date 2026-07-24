#!/usr/bin/env python3
"""Evaluate the synthetic gate and write the real-phase decision.

This script is deliberately conservative: it never promotes a real pilot when
the required teacher/seed coverage is absent or a mandatory metric is missing.
"""
import json, os
from pathlib import Path

PATH='output/elephant_source_graphdeco/sparse_observation_benchmark/track_aware_sparse_summary.json'
OUT='output/elephant_source_graphdeco/sparse_observation_benchmark/correspondence_gate_decision.json'

def main():
    records=json.load(open(PATH))['records']
    teachers=['body_roundness','ear_expansion','trunk_bending']
    clean=[r for r in records if r.get('consensus_method')=='subset' and r.get('views_per_track')==3 and r.get('outlier_rate')==0 and r.get('fallback')=='drop_track']
    robust=[r for r in records if r.get('consensus_method')=='subset' and r.get('views_per_track')==3 and r.get('outlier_rate')==.05 and r.get('fallback')=='drop_track']
    by={t:[r for r in robust if r['teacher']==t] for t in teachers}; clean_by={t:[r for r in clean if r['teacher']==t] for t in teachers}
    teacher_summary={}
    for t in teachers:
        rr=by[t]; cc=clean_by[t]
        teacher_summary[t]={'clean_records':len(cc),'robust_records':len(rr),
          'clean_active_cosine':None if not cc else max(float(x['report']['active']['cosine']) for x in cc),
          'robust_active_cosine':None if not rr else max(float(x['report']['active']['cosine']) for x in rr),
          'true_rejection':None if not rr else max(float(x['true_outlier_rejection_rate']) for x in rr),
          'false_rejection':None if not rr else min(float(x['clean_false_rejection_rate']) for x in rr),
          'accepted_outlier_mass':None if not rr else min(float(x['retained_outlier_mass']) for x in rr),
          'accepted_clean_mass':None if not rr else max(float(x['retained_clean_mass']) for x in rr)}
    mandatory=all(teacher_summary[t]['clean_records']>=1 and teacher_summary[t]['robust_records']>=1 for t in teachers)
    robust_pass=mandatory and all(teacher_summary[t]['robust_active_cosine'] is not None and teacher_summary[t]['robust_active_cosine']>=.85 and teacher_summary[t]['true_rejection']>=.90 and teacher_summary[t]['false_rejection']<.05 for t in teachers)
    pilot=Path('assets/prepared/big_carved_wooden_elephant_sculpture/real_pilot_blocky_to_rounded')
    target_sets={name:sum(1 for _ in (pilot/name).glob('*.png')) for name in ('sample_A','sample_B')}
    asset_ready=all(v>=8 for v in target_sets.values())
    if not robust_pass:
        status='SYNTHETIC_GATE_FAIL'; entered=False; reason='Synthetic correspondence gate failed or lacks the required complete teacher evidence.'
    else:
        entered=True
        if not asset_ready:
            status='REAL_ASSET_BLOCKED'; reason='Synthetic gate passed; real phase entered, but target_A/target_B do not contain two complete 8-view image sets.'
        else:
            status='REAL_PILOT_PARTIAL'; reason='Synthetic gate passed and real assets are present; run the real extractor/QC pipeline.'
    decision={'status':status,'real_phase_entered':entered,'silhouette_enabled':False,'mandatory_gate':{'complete_teacher_records':mandatory,'robust_pass':robust_pass,'background_zero_required':True,'d_scaling_zero_required':True},'teacher_summary':teacher_summary,'real_asset_counts':target_sets,'real_assets_complete':asset_ready,'reason':reason}
    os.makedirs(os.path.dirname(OUT),exist_ok=True); json.dump(decision,open(OUT,'w'),indent=2); print(json.dumps(decision,indent=2))
if __name__=='__main__': main()
