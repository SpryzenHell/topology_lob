from __future__ import annotations
import argparse, json, platform, time, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import topology_lob as tl
CAND=np.linspace(0,1,11); RADII=np.array([.35,.55,.8,1.1,1.5]); H=10; W=256; PURGE=20; LABEL=.0005; STRIDE=8

def fwd(log_mid,h=H):
    z=np.full(len(log_mid),np.nan); z[:-h]=log_mid[h:]-log_mid[:-h]; return z

def bundle(seed,events):
    raw=tl.make_synthetic_lob(events,10,seed); micro=tl.microstructure_features(raw,3)
    d,stat=tl.select_stationary_d(micro.log_mid.to_numpy()[:int(events*.7)-PURGE],CAND,.05,W)
    micro['fracdiff']=tl.fractional_diff_cpu(micro.log_mid.to_numpy(),d,W)
    idx=np.arange(50,events-H,STRIDE); clouds=tl.make_point_clouds(raw,idx,10)
    topo,meta=tl.persistent_features(clouds,RADII,False,-1)
    topodf=pd.DataFrame(topo,index=idx,columns=meta['feature_names']).reindex(range(events)).ffill()
    base=[c for c in micro.columns if c not in {'mid','log_mid'}]
    Xa=np.hstack([micro[base].to_numpy(float),topodf.to_numpy(float)])
    fr=fwd(micro.log_mid.to_numpy()); y=np.full(events,-1); ok=np.isfinite(fr); y[ok]=(fr[ok]>LABEL).astype(int)
    valid=np.isfinite(Xa).all(1)&ok&(y>=0); X=Xa[valid]; yy=y[valid]; ff=fr[valid]
    ts=int(len(X)*.7); tr=ts-PURGE
    return dict(raw=raw,micro=micro,topo=topo,topodf=topodf,meta=meta,d=d,stat=stat,clouds=clouds,X=X,y=yy,f=ff,tr=tr,ts=ts)

def features(b,topo,fd):
    m=b['micro']; base=[c for c in m.columns if c not in {'mid','log_mid'}]
    if not fd: base.remove('fracdiff')
    a=[m[base].to_numpy(float)]
    if topo: a.append(b['topodf'].to_numpy(float))
    Xa=np.hstack(a); fr=fwd(m.log_mid.to_numpy()); y=np.full(len(fr),-1); ok=np.isfinite(fr); y[ok]=(fr[ok]>LABEL).astype(int)
    v=np.isfinite(Xa).all(1)&ok&(y>=0); return Xa[v],y[v],fr[v]

def fit_eval(b,topo=True,fd=True,obj='focal',seed=2025):
    X,y,fr=features(b,topo,fd); ts=int(len(X)*.7); tr=ts-PURGE
    model=(tl.fit_logloss_xgb(X[:tr],y[:tr],seed,.35) if obj=='logloss' else tl.fit_focal_xgb(X[:tr],y[:tr],seed,.75,2,.35))
    m,_=tl.evaluate(model,X[ts:],y[ts:],fr[ts:]); return m

def walk(b,nfold=5):
    raw=b['raw']; m=b['micro']; top=b['topodf']; eligible=np.arange(W,len(raw)-H); n=len(eligible); initial=int(n*.45); size=int(n*.10); tr_end=initial; rows=[]
    for fold in range(1,nfold+1):
        raw_end=int(eligible[tr_end-1])+1; d,stat=tl.select_stationary_d(m.log_mid.to_numpy()[:raw_end],CAND,.05,W)
        frame=m.copy(); frame['fracdiff']=tl.fractional_diff_cpu(m.log_mid.to_numpy(),d,W); base=[c for c in frame.columns if c not in {'mid','log_mid'}]
        Xa=np.hstack([frame[base].to_numpy(float),top.to_numpy(float)]); fr=fwd(frame.log_mid.to_numpy()); y=np.full(len(raw),-1); ok=np.isfinite(fr); y[ok]=(fr[ok]>LABEL).astype(int); v=np.isfinite(Xa).all(1)&ok&(y>=0); X=Xa[v]; yy=y[v]; ff=fr[v]
        a=tr_end+PURGE; z=a+size; fm=tl.fit_focal_xgb(X[:tr_end],yy[:tr_end],2025+fold,.75,2,.35); bm=tl.fit_logloss_xgb(X[:tr_end],yy[:tr_end],2025+fold,.35)
        f,_=tl.evaluate(fm,X[a:z],yy[a:z],ff[a:z]); q,_=tl.evaluate(bm,X[a:z],yy[a:z],ff[a:z]); p=next(r['adf_pvalue'] for r in stat['candidates'] if r['d']==d)
        rows.append(dict(fold=fold,d=d,adf_p=p,focal=f['pearson_ic'],focal_rank=f['rank_ic'],logloss=q['pearson_ic'])); tr_end+=size
    return rows

def checks():
    pred=np.array([-2,-.5,0,.8,2.2]); y=np.array([0,1,1,0,1.]); dm=xgb.DMatrix(np.zeros((5,1)),label=y); g,h=tl.focal_grad_hess(pred,dm); eps=1e-5; ge=he=0
    for i in range(5):
        p=pred.copy(); q=pred.copy(); p[i]+=eps; q[i]-=eps
        num=(tl.focal_loss_value(p,y)-tl.focal_loss_value(q,y))/(2*eps)*len(y); ge=max(ge,abs(g[i]-num))
        gp,_=tl.focal_grad_hess(p,dm); gq,_=tl.focal_grad_hess(q,dm); he=max(he,abs(h[i]-(gp[i]-gq[i])/(2*eps)))
    base=tl.make_synthetic_lob(50,4,123); rej={}
    for k,fn in [('duplicate_timestamp',lambda d:d.iloc[[0,0]].copy()),('crossed_book',lambda d:d.assign(ask_price_1=d.bid_price_1-.1)),('negative_size',lambda d:d.assign(bid_size_1=-1))]:
        try: tl.validate_lob(fn(base)); rej[k]=False
        except tl.LOBValidationError: rej[k]=True
    shifted=base.copy()
    for i in range(1,5): shifted[f'bid_price_{i}']+=25; shifted[f'ask_price_{i}']+=25
    inv=float(np.max(np.abs(tl.make_point_clouds(base,np.array([10]),4)[0]-tl.make_point_clouds(shifted,np.array([10]),4)[0])))
    x=np.linspace(1,2,400); x2=x.copy(); x2[300:]+=10; c=float(np.nanmax(np.abs(tl.fractional_diff_cpu(x,.45,64)[:300]-tl.fractional_diff_cpu(x2,.45,64)[:300])))
    return dict(gradient_error=ge,hessian_error=he,gradient_pass=ge<1e-7,hessian_pass=he<1e-6,rejections=rej,translation_error=inv,causal_ffd_error=c,all_pass=all(rej.values()) and inv==0 and c<1e-12)

def render_text(path,lines):
    img=Image.new('RGB',(1400,820),(18,25,38)); d=ImageDraw.Draw(img); sans=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',30); bold=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',34); mono=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',28)
    d.rectangle((0,0,1400,78),fill=(31,41,55)); d.text((40,19),'Topology LOB — terminal run',font=bold,fill=(245,248,252)); d.text((1040,24),'raster evidence',font=sans,fill=(175,188,205)); y=118
    for line in lines: d.text((42,y),line,font=mono,fill=(232,239,246)); y+=40
    img.save(path)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--events',type=int,default=6000); ap.add_argument('--out',default='results/evidence'); ap.add_argument('--require-gtda',action='store_true'); args=ap.parse_args(); out=Path(args.out); (out/'figures').mkdir(parents=True,exist_ok=True); (out/'screenshots').mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter(); b=bundle(2025,args.events)
    if args.require_gtda and b['meta']['backend']!='giotto-tda': raise RuntimeError('Giotto-TDA required for evidence run')
    full=fit_eval(b); base=fit_eval(b,True,True,'logloss'); specs=[('micro_only',False,False,'focal'),('micro_plus_fd',False,True,'focal'),('micro_plus_topology',True,False,'focal'),('full_focal',True,True,'focal'),('full_logloss',True,True,'logloss')]
    ab=[]
    for name,t,f,o in specs: ab.append(dict(experiment=name,**fit_eval(b,t,f,o)))
    seeds=[]
    for s in [2025,2026,2027,2028,2029,2030]:
        sb=bundle(s,args.events); seeds.append(dict(seed=s,selected_d=sb['d'],**fit_eval(sb)))
    placebo_y=b['y'][:b['tr']].copy(); rng=np.random.default_rng(4242); rng.shuffle(placebo_y); X=b['X']; model=tl.fit_focal_xgb(X[:b['tr']],placebo_y,9090,.75,2,.35); pm,_=tl.evaluate(model,X[b['ts']:],b['y'][b['ts']:],b['f'][b['ts']:]); place=dict(real_pearson=full['pearson_ic'],real_rank=full['rank_ic'],placebo_pearson=pm['pearson_ic'],placebo_rank=pm['rank_ic'],placebo_roc_auc=pm['roc_auc'])
    wf=walk(b); chk=checks(); h1=b['topodf']['betti1_r1'].to_numpy(); fr=fwd(b['micro']['log_mid'].to_numpy()); ok=np.isfinite(h1)&np.isfinite(fr); rel={'h1_mean':float(np.mean(h1[ok])),'h1_median':float(np.median(h1[ok])),'h1_max':float(np.max(h1[ok])),'h1_forward_return_corr':float(np.corrcoef(h1[ok],fr[ok])[0,1])}; elapsed=time.perf_counter()-t0
    assert chk['all_pass'] and chk['gradient_pass'] and chk['hessian_pass'] and len(wf)==5
    fig,ax=plt.subplots(2,3,figsize=(15,9)); st=b['stat']['candidates']; ax[0,0].plot([r['d'] for r in st],[r['adf_pvalue'] for r in st],marker='o'); ax[0,0].axhline(.05,ls='--'); ax[0,0].set_yscale('log'); ax[0,0].set_title(f'Stationarity (d={b["d"]:.1f})'); ax[0,0].set_xlabel('d'); ax[0,0].set_ylabel('ADF p-value')
    ax[0,1].plot(RADII,np.median(b['topo'][:,:5],0),marker='o',label='H0'); ax[0,1].plot(RADII,np.median(b['topo'][:,5:10],0),marker='o',label='H1'); ax[0,1].set_title('Median Betti curves'); ax[0,1].set_xlabel('VR radius'); ax[0,1].set_ylabel('Betti count'); ax[0,1].legend()
    names=[r['experiment'] for r in ab]; xx=np.arange(len(names)); ax[0,2].bar(xx-.18,[r['pearson_ic'] for r in ab],.36,label='Pearson'); ax[0,2].bar(xx+.18,[r['rank_ic'] for r in ab],.36,label='Rank'); ax[0,2].set_xticks(xx,names,rotation=30,ha='right'); ax[0,2].set_title('Holdout ablation'); ax[0,2].set_ylabel('IC'); ax[0,2].legend()
    ax[1,0].plot([r['seed'] for r in seeds],[r['pearson_ic'] for r in seeds],marker='o',label='Pearson'); ax[1,0].plot([r['seed'] for r in seeds],[r['rank_ic'] for r in seeds],marker='o',label='Rank'); ax[1,0].axhline(0,linewidth=1); ax[1,0].set_title('Six-seed sensitivity'); ax[1,0].set_xlabel('seed'); ax[1,0].set_ylabel('IC'); ax[1,0].legend()
    ax[1,1].plot([r['fold'] for r in wf],[r['focal'] for r in wf],marker='o',label='Focal'); ax[1,1].plot([r['fold'] for r in wf],[r['logloss'] for r in wf],marker='o',label='Log-loss'); ax[1,1].axhline(0,linewidth=1); ax[1,1].set_title('Five-fold walk-forward'); ax[1,1].set_xlabel('fold'); ax[1,1].set_ylabel('Pearson IC'); ax[1,1].legend()
    ax[1,2].bar(['Real','Placebo'],[place['real_pearson'],place['placebo_pearson']]); ax[1,2].axhline(0,linewidth=1); ax[1,2].set_title('Label-permutation placebo'); ax[1,2].set_ylabel('Pearson IC'); fig.suptitle('Topology LOB — analysis and experiment dashboard',fontsize=20,fontweight='bold'); fig.tight_layout(); fig.savefig(out/'figures/experimental_validation_dashboard.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(2,3,figsize=(15,9)); ax[0,0].plot([r['seed'] for r in seeds],[r['roc_auc'] for r in seeds],marker='o',label='ROC-AUC'); ax[0,0].plot([r['seed'] for r in seeds],[r['pr_auc'] for r in seeds],marker='o',label='PR-AUC'); ax[0,0].set_title('Seed sensitivity: classification'); ax[0,0].legend()
    ax[0,1].hist(h1[ok],bins=np.arange(np.min(h1[ok]),np.max(h1[ok])+2)-.5); ax[0,1].set_title('H1 distribution'); ax[0,1].set_xlabel('Betti-1 at r=0.55'); ax[0,1].set_ylabel('cloud count')
    ax[0,2].bar(['Real IC','Placebo IC'],[place['real_pearson'],place['placebo_pearson']]); ax[0,2].axhline(0,linewidth=1); ax[0,2].set_title('Placebo control')
    ax[1,0].bar(['gradient','hessian'],[max(chk['gradient_error'],1e-14),max(chk['hessian_error'],1e-14)]); ax[1,0].set_yscale('log'); ax[1,0].set_title('Focal objective numerical error')
    ax[1,1].bar(['translation','causal FFD'],[max(chk['translation_error'],1e-14),max(chk['causal_ffd_error'],1e-14)]); ax[1,1].set_yscale('log'); ax[1,1].set_title('Invariance / causality error')
    ax[1,2].bar(['F-Pearson','F-Rank','L-Pearson','L-Rank'],[full['pearson_ic'],full['rank_ic'],base['pearson_ic'],base['rank_ic']]); ax[1,2].axhline(0,linewidth=1); ax[1,2].set_title('Full-stack holdout IC'); fig.suptitle('Topology LOB — robustness and correctness dashboard',fontsize=20,fontweight='bold'); fig.tight_layout(); fig.savefig(out/'figures/robustness_dashboard.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(2,2,figsize=(12,8)); m=b['micro']; ax[0,0].hist(m['spread_bps'],bins=40); ax[0,0].set_title('Spread distribution'); ax[0,0].set_xlabel('spread (bps)'); ax[0,0].set_ylabel('events'); ax[0,1].hist(m['imbalance'],bins=40); ax[0,1].set_title('Book imbalance distribution'); ax[0,1].set_xlabel('imbalance'); ax[0,1].set_ylabel('events'); ax[1,0].hist(m['rolling_vol'][np.isfinite(m['rolling_vol'])],bins=40); ax[1,0].set_title('Rolling volatility distribution'); ax[1,0].set_xlabel('rolling log-return std'); ax[1,0].set_ylabel('events'); ax[1,1].scatter(h1[ok],fr[ok],s=5,alpha=.15); ax[1,1].set_title('H1 vs 10-event forward return'); ax[1,1].set_xlabel('Betti-1 at r=0.55'); ax[1,1].set_ylabel('forward log return'); fig.suptitle('Topology LOB — synthetic data diagnostics',fontsize=20,fontweight='bold'); fig.tight_layout(); fig.savefig(out/'figures/data_diagnostics_dashboard.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    render_text(out/'screenshots/terminal_demo_run.png',[f'$ python scripts/generate_evidence.py --events {args.events} --out results/evidence',f'events:            {args.events:,}',f'model rows:        {len(X):,}',f'train rows:        {b["tr"]:,}',f'test rows:         {len(X)-b["ts"]:,}',f'purge gap:         {b["ts"]-b["tr"]}',f'TDA backend:       {b["meta"]["backend"]}',f'fractional d:      {b["d"]:.1f}',f'Focal Pearson IC:  {full["pearson_ic"]:+.6f}',f'Focal Rank IC:     {full["rank_ic"]:+.6f}',f'Log-loss Pearson:  {base["pearson_ic"]:+.6f}',f'Placebo Pearson:   {place["placebo_pearson"]:+.6f}',f'grad max error:    {chk["gradient_error"]:.2e}',f'hess max error:    {chk["hessian_error"]:.2e}',f'elapsed:            {elapsed:.2f} s','PASS: extended evidence generation and validation'])
    img=Image.new('RGB',(1650,1060),(248,250,252)); d=ImageDraw.Draw(img); sans=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',23); bold=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',34); mono=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',20); d.text((60,42),'Topology LOB — experimental validation report',font=bold,fill=(20,28,40)); d.text((60,88),'Deterministic synthetic L2 data; engineering validation, not market evidence.',font=sans,fill=(78,94,112))
    cards=[('events',f'{args.events:,}'),('model rows',f'{len(X):,}'),('selected d',f'{b["d"]:.1f}'),('TDA',b['meta']['backend']),('Focal Pearson IC',f'{full["pearson_ic"]:.4f}'),('Focal Rank IC',f'{full["rank_ic"]:.4f}')]
    for i,(k,v) in enumerate(cards): x=60+(i%3)*510; y=145+(i//3)*112; d.rounded_rectangle((x,y,x+460,y+82),12,outline=(202,213,225),width=2,fill=(255,255,255)); d.text((x+20,y+12),k,font=sans,fill=(82,98,116)); d.text((x+20,y+45),v,font=mono,fill=(20,28,40))
    y=350; d.text((60,y),'Holdout comparison',font=bold,fill=(20,28,40)); y+=58; xs=[60,410,620,830,1040,1260]; hs=['variant','Pearson IC','Rank IC','ROC-AUC','PR-AUC','log loss']
    for x,h in zip(xs,hs): d.text((x,y),h,font=bold,fill=(82,98,116))
    y=438; d.line((60,y,1540,y),fill=(202,213,225),width=2); y+=18
    for r in ab:
        vals=[r['experiment'],f'{r["pearson_ic"]:.4f}',f'{r["rank_ic"]:.4f}',f'{r["roc_auc"]:.4f}',f'{r["pr_auc"]:.4f}',f'{r["logloss"]:.4f}']
        for x,v in zip(xs,vals): d.text((x,y),v,font=mono,fill=(20,28,40))
        y+=32
    y+=12; d.text((60,y),'Walk-forward Pearson IC',font=bold,fill=(20,28,40)); y+=43; d.text((60,y),'   '.join(f'F{r["fold"]}: {r["focal"]:+.4f}' for r in wf),font=mono,fill=(20,28,40)); y+=52; d.text((60,y),'Seed sensitivity — full Focal',font=bold,fill=(20,28,40)); y+=43; d.text((60,y),'   '.join(f'{r["seed"]}: {r["pearson_ic"]:+.4f}' for r in seeds),font=mono,fill=(20,28,40)); y+=62; d.text((60,y),'Correctness gates',font=bold,fill=(20,28,40)); y+=43
    for line in [f'PASS  gradient error <= 1e-7: {chk["gradient_error"]:.2e}',f'PASS  hessian error <= 1e-6: {chk["hessian_error"]:.2e}','PASS  malformed L2 rejection / translation invariance / causal FFD','PASS  evidence rendered as embedded raster PNGs']: d.text((60,y),line,font=mono,fill=(20,28,40)); y+=34
    img.save(out/'screenshots/report_preview.png')
    res={'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'xgboost':xgb.__version__,'matplotlib':matplotlib.__version__},'main':{'events':args.events,'model_rows':len(X),'train_rows':b['tr'],'test_rows':len(X)-b['ts'],'purge_gap':b['ts']-b['tr'],'selected_d':b['d'],'tda_backend':b['meta']['backend'],'focal':full,'logloss':base,'topology_clouds':len(b['clouds']),'run_seconds':elapsed},'ablation':ab,'seed_sensitivity':seeds,'walk_forward':wf,'placebo':place,'checks':chk,'topology_relation':rel}
    (out/'experimental_validation.json').write_text(json.dumps(res,indent=2,default=lambda o:o.item() if hasattr(o,'item') else str(o)))
    (out/'EXPERIMENTAL_VALIDATION.md').write_text('# Experimental validation record\n\nAll results are deterministic synthetic-data engineering validation; they are not market-data claims.\n\n## Main run\n\n'+f'Events: **{args.events:,}**; model rows: **{len(X):,}**; train/test: **{b["tr"]:,}/{len(X)-b["ts"]:,}**; purge: **{b["ts"]-b["tr"]}**; selected d: **{b["d"]:.1f}**; TDA backend: **{b["meta"]["backend"]}**.\n\n| Metric | Focal | Log-loss control |\n|---|---:|---:|\n| Pearson IC | '+f'{full["pearson_ic"]:.6f}'+' | '+f'{base["pearson_ic"]:.6f}'+' |\n| Rank IC | '+f'{full["rank_ic"]:.6f}'+' | '+f'{base["rank_ic"]:.6f}'+' |\n| ROC-AUC | '+f'{full["roc_auc"]:.6f}'+' | '+f'{base["roc_auc"]:.6f}'+' |\n| PR-AUC | '+f'{full["pr_auc"]:.6f}'+' | '+f'{base["pr_auc"]:.6f}'+' |\n| Log loss | '+f'{full["logloss"]:.6f}'+' | '+f'{base["logloss"]:.6f}'+' |\n\n## Ablation\n\n| Variant | Pearson IC | Rank IC | ROC-AUC | PR-AUC | Log loss |\n|---|---:|---:|---:|---:|---:|\n'+'\n'.join(f'| {r["experiment"]} | {r["pearson_ic"]:.6f} | {r["rank_ic"]:.6f} | {r["roc_auc"]:.6f} | {r["pr_auc"]:.6f} | {r["logloss"]:.6f} |' for r in ab)+'\n\n## Seed sensitivity\n\n'+'\n'.join(f'- {r["seed"]}: Pearson IC {r["pearson_ic"]:+.6f}, Rank IC {r["rank_ic"]:+.6f}, d={r["selected_d"]:.1f}' for r in seeds)+'\n\n## Walk-forward\n\n'+'\n'.join(f'- Fold {r["fold"]}: d={r["d"]:.1f}, ADF p={r["adf_p"]:.4g}, Focal Pearson={r["focal"]:+.6f}, Log-loss Pearson={r["logloss"]:+.6f}' for r in wf)+'\n\n## Placebo and correctness\n\n'+f'- Label-permutation placebo Pearson IC: **{place["placebo_pearson"]:+.6f}**.\n- Gradient max error: **{chk["gradient_error"]:.3e}**.\n- Hessian max error: **{chk["hessian_error"]:.3e}**.\n- Malformed L2 rejection: **{chk["rejections"]}**.\n- Price-translation error: **{chk["translation_error"]:.3e}**.\n- Causal FFD prefix error: **{chk["causal_ffd_error"]:.3e}**.\n')
    print(json.dumps(res,indent=2,default=lambda o:o.item() if hasattr(o,'item') else str(o)))
if __name__=='__main__': main()
