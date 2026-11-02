"""Exploratory Codex usage-report forecasting, NOT server-arrival forecasting.
No tuning; fixed May 2026 calendar window and 60/20/20 time split.
Uses installed sklearn HistGradientBoostingRegressor, not the PRD's XGBoost.
"""
from pathlib import Path
from collections import defaultdict
import json, platform, time
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
START=int(pd.Timestamp('2026-05-01',tz='UTC').timestamp())
END=int(pd.Timestamp('2026-06-01',tz='UTC').timestamp())
H=10; STEP=60
TRAIN=START+int((END-START)/60*.6)*60
TEST=START+int((END-START)/60*.8)*60
d=pd.read_parquet(OUT/'calls_metadata.parquet')
d=d[d.provider.eq('codex')].sort_values('usage_report',kind='stable').reset_index(drop=True)
assert not d.duplicated(['provider','session_id','round_id']).any()
assert d.usage_report.notna().all()
assert (d[['input_tokens_total','output_tokens']]>=0).all().all()
suspect=d[d.duplicated(['user','round_id'],False)]
pilot_rows=d[d.usage_report.ge(START)&d.usage_report.lt(END)]
affected_sessions=set(suspect.session_id)&set(pilot_rows.session_id)
identity_caveat={'cross_session_round_id_extra_rows_full_codex':int(d.duplicated(['user','round_id']).sum()),
 'suspect_rows_reported_in_pilot_window':int((suspect.usage_report.ge(START)&suspect.usage_report.lt(END)).sum()),
 'pilot_sessions_with_earlier_suspect_reports':len(affected_sessions),
 'pilot_rows_in_these_sessions':int(pilot_rows.session_id.isin(affected_sessions).sum())}
# Assumption: event timestamps equal local availability; collector/ingestion latency unknown.
ix=np.arange(START//60-60,END//60,dtype=np.int64)
d['minute']=(d.usage_report//60).astype('int64')
m=d.groupby('minute').agg(reports=('round_id','size'),input=('input_tokens_total','sum'),
 output=('output_tokens','sum'),prefix=('prefix_tokens','sum'),zero_input=('input_tokens_total',lambda a:(a==0).sum())).reindex(ix,fill_value=0)
f=pd.DataFrame(index=m.index)
A=[]; B=[]; C=[]
for col in ['reports','input','output']:
    for lag in [1,2,3]:
        k=f'{col}_lag_{lag}'; f[k]=m[col].shift(lag); A.append(k)
    for w in [5,15,30,60]:
        k=f'{col}_sum_{w}'; f[k]=m[col].shift(1).rolling(w).sum(); A.append(k)
    k=f'{col}_std_15'; f[k]=m[col].shift(1).rolling(15).std(ddof=0); A.append(k)
for col in ['prefix','zero_input']:
    k=f'{col}_sum_15'; f[k]=m[col].shift(1).rolling(15).sum(); A.append(k)
origin=m.index.to_numpy()*60
f['hour_sin']=np.sin(2*np.pi*(origin%86400)/86400)
f['hour_cos']=np.cos(2*np.pi*(origin%86400)/86400)
A+=['hour_sin','hour_cos']
models=sorted(d.loc[d.usage_report<TRAIN,'model'].dropna().unique())
for model in models:
    by=d[d.model.eq(model)].groupby('minute').size().reindex(ix,fill_value=0)
    k='model_share_'+str(len(A)); f[k]=by.shift(1).rolling(15).sum()/f.reports_sum_15.replace(0,np.nan); A.append(k)
times=d.usage_report.to_numpy()
inputs=d.input_tokens_total.to_numpy()
origins=np.arange(START,END,60,dtype=np.int64)
dist=[]
for t in origins:
    l,r=np.searchsorted(times,[t-900,t],side='left'); x=inputs[l:r]
    dist.append([float(x.mean()),float(x.std()),float(np.median(x)),float(np.quantile(x,.9))] if len(x) else [np.nan]*4)
distkeys=['input_len_mean_15','input_len_std_15','input_len_median_15','input_len_p90_15']
for k,v in zip(distkeys,np.asarray(dist).T): f.loc[origins//60,k]=v
A+=distkeys
# Strictly replay usage reports; no final session lifetime, final counts or future tool results.
state={}; ptr=0; state_rows=[]; max_used=[]
rs=list(d[['usage_report','session_id','input_tokens_total']].itertuples(index=False,name=None))
for t in origins:
    while ptr<len(rs) and rs[ptr][0]<t:
        tm,sid,inp=rs[ptr]
        old=state.get(sid)
        state[sid]=(tm,old[1] if old else tm,(old[2]+1) if old else 1,inp,(inp-old[3]) if old else np.nan)
        ptr+=1
    active=[s for s in state.values() if s[0]>=t-1800]
    counts=[sum(s[0]>=t-w*60 for s in active) for w in [5,15,30]]
    v=[]
    for vals in [[t-s[0] for s in active],[t-s[1] for s in active],
                 [s[2] for s in active],[s[3] for s in active],[s[4] for s in active]]:
        a=np.asarray(vals,dtype=float); a=a[np.isfinite(a)]
        v.extend([float(np.median(a)),float(np.quantile(a,.9))] if len(a) else [np.nan,np.nan])
    state_rows.append(counts+v)
    max_used.append(rs[ptr-1][0] if ptr else np.nan)
B=[f'recent_reporting_sessions_{w}' for w in [5,15,30]]
C=[name+'_'+q for name in ['session_recency_seconds','session_observed_age_seconds',
 'session_observed_report_count','session_last_reported_input','session_reported_input_change'] for q in ['median','p90']]
for k,v in zip(B+C,np.asarray(state_rows).T): f.loc[origins//60,k]=v
f=f.loc[origins//60,A+B+C].copy()
f.insert(0,'origin_utc',pd.to_datetime(origins,unit='s',utc=True))
f['label_available_at_utc']=pd.to_datetime(origins+H*60,unit='s',utc=True)
f['max_report_used']=max_used
for col in ['input','output']:
    cs=np.r_[0,m[col].cumsum()]
    pos=np.searchsorted(m.index.to_numpy(),f.index.to_numpy())
    valid=pos+H<=len(m)
    f['target_'+col]=np.where(valid,cs[np.minimum(pos+H,len(m))]-cs[pos],np.nan)
    f['last10_'+col]=m[col].shift(1).rolling(10).sum().reindex(f.index)
    f['mean60_'+col]=m[col].shift(1).rolling(60).mean().reindex(f.index)*10
f=f[f.target_input.notna() & (origins+H*60<=END)].copy()
t=f.origin_utc.astype('int64').to_numpy()//10**9
f['split']=np.where((t<TRAIN)&(t+600<=TRAIN),'train',np.where((t>=TRAIN)&(t<TEST)&(t+600<=TEST),'validation',np.where(t>=TEST,'test','purged')))
assert (f.max_report_used < t).all()
assert ((f.loc[f.split.eq('train'),'label_available_at_utc'].astype('int64')//10**9)<=TRAIN).all()
assert ((f.loc[f.split.eq('validation'),'label_available_at_utc'].astype('int64')//10**9)<=TEST).all()
# Independently sum source rows and rebuild session statistics at sampled origins.
checks=[]
for idx in np.linspace(0,len(f)-1,9).astype(int):
    row=f.iloc[idx]; tt=int(row.origin_utc.timestamp())
    future=d[(d.usage_report>=tt)&(d.usage_report<tt+600)]
    assert int(future.input_tokens_total.sum())==row.target_input
    assert int(future.output_tokens.sum())==row.target_output
    past=d[(d.usage_report>=tt-900)&(d.usage_report<tt)]
    assert len(past)==row.reports_sum_15
    assert int(past.input_tokens_total.sum())==row.input_sum_15
    history=d[d.usage_report<tt]
    lasts=history.groupby('session_id').tail(1)
    assert int((lasts.usage_report>=tt-1800).sum())==row.recent_reporting_sessions_30
    act=lasts[lasts.usage_report>=tt-1800]
    if len(act): assert np.isclose(np.median(tt-act.usage_report),row.session_recency_seconds_median)
    checks.append(tt)
f.to_parquet(OUT/'pilot_forecast_table.parquet',index=False)
m.to_parquet(OUT/'pilot_minute_reports.parquet')
config={'scope':'Exploratory CODEx usage-report clock; not original arrival-demand task, not actual per-minute generated tokens.',
 'window_utc':['2026-05-01T00:00:00Z','2026-06-01T00:00:00Z'],'horizon_minutes':10,'step_seconds':60,
 'train_cutoff_utc':pd.to_datetime(TRAIN,unit='s',utc=True).isoformat(),'test_start_utc':pd.to_datetime(TEST,unit='s',utc=True).isoformat(),
 'model':'sklearn HistGradientBoostingRegressor','params':dict(loss='absolute_error',learning_rate=.06,max_iter=100,max_leaf_nodes=15,
  min_samples_leaf=40,l2_regularization=1.0,max_bins=63,early_stopping=False,random_state=42),
 'tuning_trials':0,'model_fit_count':6,'refit_on_validation':False,
 'features':{'A':A,'B_extra':B,'C_extra':C},'model_share_mapping':dict(zip([k for k in A if k.startswith('model_share')],models)),
 'availability':'All feature usage reports strictly < origin; input length first used at report. Labels known after window end under timestamp-as-local-availability assumption. Ingestion delay unknown.',
 'session_definition':'Recently reporting session: at least one report in trailing W min; not proof an agent is currently running. IDs are join keys only.',
 'C_scope':'Observed per-session usage state only. No true tool-wait, turn-end, or running status features in this pilot.',
 'split_counts':f.split.value_counts().to_dict(),'features_count':{'A':len(A),'B':len(A+B),'C':len(A+B+C)},
 'cross_session_identity_caveat':identity_caveat,
 'checks':{'raw_target_and_history_and_session_spot_checks':len(checks),'strict_feature_availability':True,
   'training_labels_ready_by_fit':True,'overlapping_train_validation_test_targets_purged':True},
 'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},
 'limitations':['Full-period EDA seen before pilot; not pristine held-out confirmatory evaluation.',
 'Missing logs versus no activity cannot be distinguished; zeros mean no recorded report, not verified zero demand.',
 'Historical donations and participant/model composition shift limit external validity.',
 'Six calendar days of test with overlapping 10-min targets; no independent-minute significance claim.',
 'C tests session usage summaries, not observed execution/tool states; XGBoost not installed.',
 'Two May sessions have earlier April cross-session round-ID repeats; historical C counts/age may be affected. No suspect repeat rows fall inside May targets.']}
periods=[]
for split,g in f[f.split.ne('purged')].groupby('split',sort=False):
    l=int(g.origin_utc.min().timestamp()); r=int(g.origin_utc.max().timestamp())+600
    v=d[(d.usage_report>=l)&(d.usage_report<r)]
    periods.append(dict(split=split,origins=len(g),reports=len(v),users=v.user.nunique(),sessions=v.session_id.nunique(),
     models=v.model.nunique(),zero_input_target_pct=float((g.target_input==0).mean()*100),
     largest_user_report_share=float(v.user.value_counts(normalize=True).iloc[0]),largest_model_report_share=float(v.model.value_counts(normalize=True).iloc[0])))
pd.DataFrame(periods).to_csv(OUT/'pilot_periods.csv',index=False)
# Horizon density is descriptive, not selection by model error.
horizons=[]
period=m.loc[START//60:END//60-1]
for h in [1,5,10]:
    cs=np.r_[0,period.reports.cumsum()]; y=cs[h:]-cs[:-h]
    horizons.append({'minutes':h,'origins':len(y),'zero_report_target_pct':float((y==0).mean()*100)})
config['horizon_density_on_pilot_window']=horizons
(OUT/'pilot_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False))
metrics=[]; daily=[]; predictions=[]
def score(actual,pred):
    e=actual-pred
    return {'n':len(actual),'mae':float(np.abs(e).mean()),'rmse':float(np.sqrt((e**2).mean()))}
train=f.split.eq('train')
for target in ['input','output']:
    y=f['target_'+target].to_numpy(); threshold=float(np.quantile(y[train],.9))
    pred={k:f[k+'_'+target].to_numpy() for k in ['last10','mean60']}
    for name,cols in [('A',A),('B',A+B),('C',A+B+C)]:
        model=HistGradientBoostingRegressor(**config['params'])
        model.fit(f.loc[train,cols],y[train]); pred[name]=np.maximum(0,model.predict(f[cols]))
    for split in ['validation','test']:
        sm=f.split.eq(split).to_numpy()
        for name,yp in pred.items():
            slices={'all':sm,'zero':sm&(y==0),'positive':sm&(y>0),'high_train_p90':sm&(y>=threshold),
                    'nonoverlap_10min':sm&((t[:len(f)]-START)%600==0)}
            for sl,mask in slices.items():
                if mask.any(): metrics.append(dict(target=target,split=split,model=name,slice=sl,**score(y[mask],yp[mask])))
            day=f.loc[sm,['origin_utc']].copy();day['abs_error']=np.abs(y[sm]-yp[sm]); day['date_utc']=day.origin_utc.dt.strftime('%Y-%m-%d')
            for date,g in day.groupby('date_utc'): daily.append(dict(target=target,split=split,model=name,date_utc=date,n=len(g),mae=float(g.abs_error.mean())))
            if split=='test': predictions.append(pd.DataFrame({'origin_utc':f.loc[sm,'origin_utc'].to_numpy(),'target':target,'model':name,'actual':y[sm],'prediction':yp[sm]}))
met=pd.DataFrame(metrics);met.to_csv(OUT/'pilot_metrics.csv',index=False)
pd.DataFrame(daily).to_csv(OUT/'pilot_daily_mae.csv',index=False)
pd.concat(predictions).to_parquet(OUT/'pilot_predictions.parquet',index=False)
selected={}
for target in ['input','output']:
    v=met[(met.target==target)&(met.split=='validation')&(met.slice=='all')&met.model.isin(['A','B','C'])]
    selected[target]=v.loc[v.mae.idxmin(),'model']
config['selected_by_validation_mae']=selected
(OUT/'pilot_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False))
print(met[(met.slice=='all')].to_string(index=False))
print('Selected using validation only:',selected)
print('Checks:',config['checks'],'Split counts:',config['split_counts'])
