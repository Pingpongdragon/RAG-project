#!/usr/bin/env python3
"""Unified paper-figure generator (plotting/plot_main.py).
Run from any directory:  python plotting/plot_main.py
JSON stores recall@5 and kb_coverage already in % (0-100 scale).
"""
from __future__ import annotations
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT  = Path(__file__).parent.parent  # motivation/ root
FIGS  = ROOT / 'paper_figs'
FIGS.mkdir(exist_ok=True)

MO1_SUDDEN  = ROOT.parent / 'experiments/direct/data/results_100w_sudden_v3.json'
MO1_GRADUAL = ROOT.parent / 'experiments/direct/data/results_100w_gradual_v3_instrumented.json'
MO2_SUDDEN  = ROOT.parent / 'experiments/hidden/data/results_50w_sudden_v3.json'
MO2_GRADUAL = ROOT.parent / 'experiments/hidden/data/results_50w_gradual_v3.json'

plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':11,
    'axes.labelsize':12,'axes.titlesize':11,
    'legend.fontsize':9,'xtick.labelsize':9.5,'ytick.labelsize':9.5,
    'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'ps.fonttype':42,
})
PRE_BG='#EDF4FC'; POST_BG='#FDF5E6'

# ── Production-equivalent online-serving latency model ─────────────────
# Hot-tier KB sits in-memory (FAISS, sub-100ms). In production the cold
# pool would be a remote managed vector DB (Pinecone/Milvus/Qdrant); its
# end-to-end p50 over a 200K-doc index is reported at 30-80 ms by
# VectorDBBench (Zilliz, 2024) and Pinecone serverless docs, vs ~5-15 ms
# for in-process FAISS on the same data — i.e. ~3-8x. We therefore scale
# the locally-measured cold-tier search by COLD_REMOTE_SCALE; we do NOT
# add a separate RTT constant (network/serialisation is already absorbed
# in the published p50 numbers we calibrate against).
COLD_REMOTE_SCALE = 5.0  # mid-point of 3-8x range

def _hot_serve_ms(m):
    """Hot-tier (in-memory FAISS) KB-search latency per query, ms."""
    if 'retrieval_latency_mean_ms' in m:
        return float(m['retrieval_latency_mean_ms'])
    return float(m.get('query_latency_mean_ms', 0.0)) \
        - float(m.get('prepare_latency_mean_ms', 0.0))

# Window size used by all motivation runs (50 queries per window).
# Raw `*_latency_mean_ms` fields are summed over the whole window; we
# divide by this constant to report production-style per-query latency.
WINDOW_SIZE_FOR_LATENCY = 50

def serve_latency_ms(m):
    """Production-equivalent online serving latency per QUERY (ms).

    Raw retrieval/prepare timings are window-totals over WINDOW_SIZE_FOR_LATENCY
    queries; we normalise to per-query and add cold-tier RTT (paid once per
    window of OnDemand misses, batch-amortised).

    Cache / Static / QD / Oracle: hot-tier KB search only (in-memory FAISS).
    OnDemandFetch: hot-tier check + remote cold-tier search
                   = (local cold-search * COLD_REMOTE_SCALE)/window + RTT_amortised.
    """
    ws = WINDOW_SIZE_FOR_LATENCY
    hot_q = max(_hot_serve_ms(m), 0.0) / ws            # ms / query
    cold_local_win = float(m.get('prepare_latency_mean_ms', 0.0))  # ms / window
    if cold_local_win > 1.0:
        # OnDemand pays the (scaled) cold-tier search per miss.
        # Per-query: hot KB search + remote cold-tier search (FAISS x scale).
        return hot_q + (cold_local_win * COLD_REMOTE_SCALE) / ws
    return hot_q

# (color, linestyle)
PALETTE = {
    'Static':             ('#9E9E9E',(0,(2,2))),
    'DocArrival':         ('#43A047',(0,(6,2))),
    'RandomFIFO':         ('#BCBD22',(0,(3,2))),
    'KnowledgeEdit':      ('#8E24AA',(0,(5,2,1,2))),
    'LRU':                ('#F4511E',(0,(4,2))),
    'GPTCacheStyle':      ('#E91E63',(0,(3,1,1,1))),
    'MemGPTStyle':        ('#FF9800',(0,(5,1))),
    'TemporalAware':      ('#2563EB',(0,(4,1,1,1))),
    'OnDemandFetch':      ('#00ACC1',(0,(2,1.5))),
    'LogDrivenArrival':   ('#EF8C00',(0,(7,2))),
    'DRIP-Dense': ('#1565C0','-'),
    'DRIP-Dense-Loose':   ('#0288D1',(0,(10,2))),
    'Oracle':             ('#C62828','-'),
    # Mo2-only
    'RandomFIFO':         ('#BCBD22',(0,(3,2))),
    'OnDemandFetch':      ('#00ACC1',(0,(2,1.5))),
    'LogDrivenArrival':   ('#EF8C00',(0,(7,2))),
}
LABELS={
    'Static':'Static (frozen)','DocArrival':'Doc-Arrival',
    'RandomFIFO':'Random-FIFO','KnowledgeEdit':'KnowledgeEdit',
    'LRU':'LRU (fair)','GPTCacheStyle':'GPTCache-style (fair)',
    'MemGPTStyle':'MemGPT-style (fair)','TemporalAware':'Temporal-Aware (Temporal-RAG)',
    'OnDemandFetch':'On-Demand (per-query)',
    'LogDrivenArrival':'Log-Driven (lagged)',
    'DRIP-Dense':r'NQM (ours)',
    'DRIP-Dense-Loose':'DRIP-Dense-Loose','Oracle':'Oracle',
    'RandomFIFO':'Random-FIFO','OnDemandFetch':'On-Demand (per-query)',
    'LogDrivenArrival':'Log-Driven (lagged)',
}
DS_TITLES_MO1={'hotpotqa_comparison':'HotpotQA-comp','2wiki_simple':'2wiki-simple',
               'fever':'FEVER','triviaqa_wikipedia':'TriviaQA-Wiki'}
DS_TITLES_MO2={'hotpotqa':'HotpotQA','2wikimultihopqa':'2WikiMultihopQA','musique':'MuSiQue'}
SHOW_MAIN=['Static','DocArrival','KnowledgeEdit','LRU','GPTCacheStyle','MemGPTStyle',
           'OnDemandFetch','DRIP-Dense','Oracle']
# Mo2 has extra strategies: RandomFIFO, OnDemandFetch, LogDrivenArrival
SHOW_MAIN_MO2=['Static','RandomFIFO','DocArrival','KnowledgeEdit','LRU',
               'GPTCacheStyle','MemGPTStyle','OnDemandFetch','LogDrivenArrival',
               'DRIP-Dense','Oracle']
SHOW_FULL=list(PALETTE.keys())
SHOW_FULL_MO2=list(PALETTE.keys()) + ['RandomFIFO','OnDemandFetch','LogDrivenArrival']

def smooth(arr, half, w=6):
    k=np.exp(-0.5*(np.arange(-(w//2),w//2+1)/(w/3.0))**2); k/=k.sum(); p=w//2
    def c(s): return np.convolve(np.pad(s,p,mode='edge'),k,mode='valid')
    return np.concatenate([c(arr[:half]),c(arr[half:])])

def draw_panel(ax, res, show, mk, ylabel, lw1=3.0, lw2=2.4, lw3=1.6, sw=6, first=False):
    cfg=res['config']; nw=cfg['n_windows']; half=nw//2; x=np.arange(1,nw+1)
    ax.axvspan(0.5,half+0.5,facecolor=PRE_BG,alpha=1.0,zorder=0)
    ax.axvspan(half+0.5,nw+0.5,facecolor=POST_BG,alpha=1.0,zorder=0)
    for name in show:
        if name not in res['summary']: continue
        per=res['summary'][name].get(mk)
        if not per: continue
        col,ls=PALETTE.get(name,('#888','-'))
        # values already in % (0–100 scale)
        vals=smooth(np.asarray(per, dtype=float), half, sw)
        if name=='DRIP-Dense': lw,al,z=lw1,1.0,7
        elif name=='Oracle':           lw,al,z=lw2,0.95,6
        elif name in ('GPTCacheStyle','LRU','MemGPTStyle'): lw,al,z=lw3,0.9,5
        elif name=='Static':           lw,al,z=lw3,0.85,3
        else:                          lw,al,z=lw3,0.75,2
        ax.plot(x, vals, color=col, linestyle=ls, linewidth=lw, alpha=al, zorder=z,
                label=LABELS.get(name,name),solid_capstyle='round',dash_capstyle='round')
    ax.axvline(half+0.5,color='#333',ls='-',lw=1.4,alpha=0.6,zorder=8)
    if first:
        ax.text(half+0.5,103,'drift onset',ha='center',va='bottom',fontsize=10,
                fontweight='bold',color='#333',
                bbox=dict(boxstyle='round,pad=0.22',fc='white',ec='#333',lw=1.0,alpha=0.92))
        ax.text(half*0.5,3,'pre-drift',ha='center',va='bottom',fontsize=9.5,
                color='#3A5F8A',fontweight='bold',alpha=0.95)
        ax.text(half+(nw-half)*0.5,3,'post-drift',ha='center',va='bottom',
                fontsize=9.5,color='#9C5A00',fontweight='bold',alpha=0.95)
    ax.set_xlim(0.5,nw+0.5); ax.set_ylim(0,107)
    ax.set_ylabel(ylabel,fontsize=12,fontweight='bold')
    ax.grid(True,axis='y',alpha=0.20,linestyle=':',zorder=1)

def build_fig(data, ds_map, title, show, fw, fh, lnc=4, sw=6, suffix=''):
    datasets=list(ds_map.keys()); ncols=len(datasets)
    mks=['cov_per_window','recall@5_per_window']
    ylbls=['KB Coverage (%)','Recall@5 (%)']
    fig,axes=plt.subplots(2,ncols,figsize=(fw,fh),sharex='col')
    if ncols==1: axes=axes.reshape(2,1)
    for row,(mk,ylbl) in enumerate(zip(mks,ylbls)):
        for col,ds in enumerate(datasets):
            ax=axes[row,col]
            if ds not in data: ax.set_visible(False); continue
            res=data[ds]
            draw_panel(ax,res,show,mk,ylbl,sw=sw,first=(row==0 and col==0))
            if row==0:
                cfg=res['config']
                ax.set_title(f'{ds_map[ds]}\npool={cfg["pool_size"]:,}, KB={cfg["kb_budget"]:,}',
                             fontsize=11,fontweight='bold',pad=7)
            if row==len(mks)-1: ax.set_xlabel('Window index (time →)',fontsize=11)
    h,l=axes[0,0].get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',ncol=lnc,frameon=False,
               bbox_to_anchor=(0.5,-0.01),fontsize=10,handlelength=3.5,
               columnspacing=1.4,handletextpad=0.6)
    fig.suptitle(title+suffix,fontsize=11.5,fontweight='bold',y=1.005)
    fig.tight_layout(rect=[0,0.08,1,1])
    return fig


def build_fig_1row(data, ds_map, title, show, fw, fh, lnc=4, sw=6, suffix=''):
    """Recall-only single-row figure — for Mo2 which has no cov_per_window data."""
    datasets=list(ds_map.keys()); ncols=len(datasets)
    fig,axes=plt.subplots(1,ncols,figsize=(fw,fh),sharex=False)
    if ncols==1: axes=[axes]
    for col,ds in enumerate(datasets):
        ax=axes[col]
        if ds not in data: ax.set_visible(False); continue
        res=data[ds]
        draw_panel(ax,res,show,'recall@5_per_window','Recall@5 (%)',sw=sw,first=(col==0))
        cfg=res['config']
        ax.set_title(f'{ds_map[ds]}\npool={cfg["pool_size"]:,}, KB={cfg["kb_budget"]:,}',
                     fontsize=11,fontweight='bold',pad=7)
        ax.set_xlabel('Window index (time →)',fontsize=11)
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',ncol=lnc,frameon=False,
               bbox_to_anchor=(0.5,-0.01),fontsize=10,handlelength=3.5,
               columnspacing=1.4,handletextpad=0.6)
    fig.suptitle(title+suffix,fontsize=11.5,fontweight='bold',y=1.005)
    fig.tight_layout(rect=[0,0.08,1,1])
    return fig

def save(fig, stem):
    for ext in ('pdf','png'):
        p=FIGS/f'{stem}.{ext}'
        fig.savefig(p,bbox_inches='tight',dpi=(None if ext=='pdf' else 200))
        print(f'  Saved {p}')
    plt.close(fig)

def load(path):
    return json.load(open(path))

def h2v(rd, ds, s):
    return rd.get(ds,{}).get('summary',{}).get(s,{}).get('recall@5_h2', None)

print('Loading results…')
mo1s=load(MO1_SUDDEN); mo1g=load(MO1_GRADUAL)
# Hot-tier cache framing: replace HotpotQA-comp with the realistic
# pool=200k experiment so pool/KB ratio (~10x) matches a production
# hot-tier semantic cache fronting a much larger cold-tier vector store.
MO1_GRADUAL_HOTPOT_HOTTIER = ROOT.parent / 'experiments/direct/data/results_hotpotqa_comp_gradual_pool200k.json'
if MO1_GRADUAL_HOTPOT_HOTTIER.exists():
    _hot = load(MO1_GRADUAL_HOTPOT_HOTTIER)
    if 'hotpotqa_comparison' in _hot:
        mo1g['hotpotqa_comparison'] = _hot['hotpotqa_comparison']
        print(f'  HotpotQA-comp overridden with hot-tier (pool={_hot["hotpotqa_comparison"]["config"]["pool_size"]:,}, KB={_hot["hotpotqa_comparison"]["config"]["kb_budget"]:,})')
# Override other mo1 datasets with expanded-pool hot-tier runs when available
for _ds_key, _fname in [
    ('fever',              'results_fever_gradual_pool100k.json'),
    ('triviaqa_wikipedia', 'results_triviaqa_gradual_pool40k.json'),
    ('2wiki_simple',       'results_2wiki_simple_gradual_pool140k.json'),
]:
    _p = ROOT.parent / f'experiments/direct/data/{_fname}'
    if _p.exists():
        _d = load(_p)
        if _ds_key in _d:
            mo1g[_ds_key] = _d[_ds_key]
            _cfg = _d[_ds_key]['config']
            print(f'  {_ds_key} overridden (pool={_cfg["pool_size"]:,}, KB={_cfg["kb_budget"]:,})')
# ── Mo1 SUDDEN hot-tier overrides (mirror of gradual block above) ──
# Replace legacy small-pool sudden runs (1.4–6.8x ratio, mixed nw) with
# rerun results matching the gradual hot-tier configs (≈10x ratio, nw=100),
# so figA1 (sudden) is comparable to figA2 (gradual).
for _ds_key, _fname in [
    ('hotpotqa_comparison', 'results_hotpotqa_comp_sudden_pool200k.json'),
    ('fever',               'results_fever_sudden_pool100k.json'),
    ('triviaqa_wikipedia',  'results_triviaqa_sudden_pool40k.json'),
    ('2wiki_simple',        'results_2wiki_simple_sudden_pool140k.json'),
]:
    _p = ROOT.parent / f'experiments/direct/data/{_fname}'
    if _p.exists():
        _d = load(_p)
        if _ds_key in _d:
            mo1s[_ds_key] = _d[_ds_key]
            _cfg = _d[_ds_key]['config']
            print(f'  [sudden] {_ds_key} overridden (pool={_cfg["pool_size"]:,}, KB={_cfg["kb_budget"]:,}, nw={_cfg["n_windows"]})')
mo2s=load(MO2_SUDDEN); mo2g=load(MO2_GRADUAL)
# Multi-hop hot-tier override: replace 2WikiMultihopQA with the larger-pool
# (pool=67k / KB=6.7k = 10x ratio) gradual run so fig2(b) matches the
# production hot-tier framing used for fig2(a).
MO2_GRADUAL_2WIKI_HOTTIER = ROOT.parent / 'experiments/hidden/data/results_100w_2wiki_gradual_pool67k.json'
if MO2_GRADUAL_2WIKI_HOTTIER.exists():
    _hot2 = load(MO2_GRADUAL_2WIKI_HOTTIER)
    if '2wikimultihopqa' in _hot2:
        mo2g['2wikimultihopqa'] = _hot2['2wikimultihopqa']
        _c2 = _hot2['2wikimultihopqa']['config']
        print(f'  2WikiMultihopQA overridden with hot-tier (pool={_c2["pool_size"]:,}, KB={_c2["kb_budget"]:,})')
# ── Mo2 SUDDEN hot-tier overrides (mirror of gradual block above) ──
# Replace legacy 50w small-pool sudden runs (1.6–2.3x ratio, nw=20–50) with
# 100w hot-tier sudden reruns matching the gradual pools.
for _ds_key, _fname in [
    ('2wikimultihopqa', 'results_100w_2wiki_sudden_pool67k.json'),
    ('hotpotqa',        'results_100w_hotpot_bridge_sudden_pool33k.json'),
    ('musique',         'results_100w_musique_sudden_pool70k.json'),
]:
    _p = ROOT.parent / f'experiments/hidden/data/{_fname}'
    if _p.exists():
        _d = load(_p)
        if _ds_key in _d:
            mo2s[_ds_key] = _d[_ds_key]
            _cfg = _d[_ds_key]['config']
            print(f'  [sudden] {_ds_key} overridden (pool={_cfg["pool_size"]:,}, KB={_cfg["kb_budget"]:,}, nw={_cfg["n_windows"]})')


# ── Curve figures ──────────────────────────────────────────────────────
print('Building curve figures…')
ENC_DENSE = 'Encoder: all-MiniLM-L6-v2 · Dense retrieval'
# Mo1 — 2 rows: KB Coverage + Recall@5
for drift, data, title in [
    ('sudden',  mo1s, f'Sudden drift — single-hop  ({ENC_DENSE})'),
    ('gradual', mo1g, f'Gradual drift — single-hop  ({ENC_DENSE})'),
]:
    fig=build_fig(data,DS_TITLES_MO1,title,SHOW_MAIN,16.0,8.4,lnc=4)
    save(fig,f'mo1_curves_{drift}')
    fig=build_fig(data,DS_TITLES_MO1,title,SHOW_FULL,16.0,8.4,lnc=4,suffix=' (appendix)')
    save(fig,f'mo1_curves_{drift}_full')

# Mo2 — 1 row: Recall@5 only (cov_per_window not recorded in Mo2 pipeline)
for drift, data, title in [
    ('sudden',  mo2s, f'Sudden drift — multi-hop  ({ENC_DENSE})'),
    ('gradual', mo2g, f'Gradual drift — multi-hop  ({ENC_DENSE})'),
]:
    fig=build_fig_1row(data,DS_TITLES_MO2,title,SHOW_MAIN_MO2,14.5,4.8,lnc=5)
    save(fig,f'mo2_curves_{drift}')
    fig=build_fig_1row(data,DS_TITLES_MO2,title,SHOW_FULL_MO2,14.5,4.8,lnc=5,suffix=' (appendix)')
    save(fig,f'mo2_curves_{drift}_full')

# ── H2 bar chart ───────────────────────────────────────────────────────
print('Building H2 bar charts…')
STRATS_BAR=['Static','DocArrival','KnowledgeEdit','LRU','GPTCacheStyle','MemGPTStyle','DRIP-Dense']
BAR_COLORS={'Static':'#9E9E9E','DocArrival':'#43A047','KnowledgeEdit':'#8E24AA',
            'LRU':'#F4511E','GPTCacheStyle':'#E91E63','MemGPTStyle':'#FF9800',
            'DRIP-Dense':'#1565C0'}
BAR_LABELS={'Static':'Static','DocArrival':'Doc-Arrival','KnowledgeEdit':'KnowledgeEdit',
            'LRU':'LRU (fair)','GPTCacheStyle':'GPTCache (fair)','MemGPTStyle':'MemGPT (fair)',
            'DRIP-Dense':'NQM (ours)'}

def h2_bar(cells, title, fname):
    n=len(cells); ns=len(STRATS_BAR); x=np.arange(n); w=0.11
    offsets=np.linspace(-(ns-1)/2,(ns-1)/2,ns)*w
    fig,ax=plt.subplots(figsize=(13,5.5))
    for si,sn in enumerate(STRATS_BAR):
        # values already in %
        vals=[rd.get(dk,{}).get('summary',{}).get(sn,{}).get('recall@5_h2',0) for _,rd,dk in cells]
        ax.bar(x+offsets[si],vals,w*0.92,color=BAR_COLORS[sn],zorder=3,label=BAR_LABELS[sn])
        if sn=='DRIP-Dense':
            for xi,v in zip(x+offsets[si],vals):
                ax.text(xi,v+0.5,f'{v:.1f}',ha='center',va='bottom',fontsize=7,
                        fontweight='bold',color='#1565C0')
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in cells],fontsize=10)
    ax.set_ylabel('Recall@5 H2 (%)',fontsize=12); ax.set_title(title,fontsize=12,fontweight='bold')
    ax.grid(True,axis='y',alpha=0.22,linestyle=':')
    ax.legend(loc='upper right',frameon=False,fontsize=9,ncol=2)
    fig.tight_layout(); return fig

for drift,rd in [('sudden',mo1s),('gradual',mo1g)]:
    cells=[(DS_TITLES_MO1[k],rd,k) for k in DS_TITLES_MO1]
    fig=h2_bar(cells,f'H2 Recall@5 — {drift} drift  (single-hop · {ENC_DENSE})',f'h2_bar_{drift}')
    save(fig,f'h2_bar_{drift}')

# ── Write-efficiency scatter ───────────────────────────────────────────
print('Building write-efficiency scatter…')
SCATTER=['Static','DocArrival','KnowledgeEdit','LRU','GPTCacheStyle','MemGPTStyle','DRIP-Dense','Oracle']
MARKERS={'Static':'o','DocArrival':'s','KnowledgeEdit':'^','LRU':'D','GPTCacheStyle':'P',
         'MemGPTStyle':'X','DRIP-Dense':'*','Oracle':'v'}
fig4,ax4=plt.subplots(figsize=(8.5,6.0))
for sn in SCATTER:
    xs,ys=[],[]
    for rd in (mo1s,mo1g):
        for dk in DS_TITLES_MO1:
            r=rd.get(dk,{}).get('summary',{}).get(sn,{})
            if r: xs.append(r.get('update_cost',0)); ys.append(r.get('recall@5_h2',0))
    if not xs: continue
    col=PALETTE.get(sn,('#888','-'))[0]
    ax4.scatter(xs,ys,s=(220 if sn=='DRIP-Dense' else 70),c=col,
                marker=MARKERS.get(sn,'o'),label=BAR_LABELS.get(sn,sn),alpha=0.82,zorder=5,
                edgecolors=('#0D47A1' if sn=='DRIP-Dense' else 'white'),linewidths=1.5)
ax4.set_xlabel('Total KB writes (update_cost)',fontsize=12)
ax4.set_ylabel('Post-drift Recall@5 H2 (%)',fontsize=12)
ax4.set_title('Write efficiency: post-drift quality vs. write cost\n'
              '(each point = dataset×drift, Mo1 only; cache baselines use passive arrival stream)',
              fontsize=11)
ax4.set_xscale('symlog',linthresh=10); ax4.grid(True,alpha=0.2,linestyle=':')
ax4.legend(frameon=False,fontsize=9,loc='lower right',ncol=2)
fig4.tight_layout(); save(fig4,'write_efficiency')

# ── LaTeX table ────────────────────────────────────────────────────────
print('Building LaTeX table…')
lines=[
    r'\begin{table*}[t]',r'\centering',r'\small',
    r'\caption{Post-drift Recall@5 H2 (\%): NQM vs fair cache baselines. '
    r'Cache baselines use passive doc-arrival only (no failure probing). '
    r'\textbf{Bold}=best non-oracle. $\Delta$=QDC minus best cache.}',
    r'\label{tab:h2_comparison}',r'\setlength{\tabcolsep}{4pt}',
    r'\begin{tabular}{llrrrrrr}',r'\toprule',
    r'Dataset & Drift & Static & LRU & GPTCache & MemGPT & \textbf{QDC (ours)} & $\Delta$ \\',r'\midrule',
]
for ds,dl in DS_TITLES_MO1.items():
    for drift,rd in [('Sudden',mo1s),('Gradual',mo1g)]:
        st=h2v(rd,ds,'Static'); lru=h2v(rd,ds,'LRU'); gpt=h2v(rd,ds,'GPTCacheStyle')
        mem=h2v(rd,ds,'MemGPTStyle'); qdc=h2v(rd,ds,'DRIP-Dense')
        if None in (st,lru,gpt,mem,qdc): continue
        best_cache=max(lru,gpt,mem)
        def fmt(v,best=best_cache):
            s=f'{v:.1f}'; return r'\textbf{'+s+'}' if abs(v-best)<0.05 else s
        delta=qdc-best_cache
        lines.append('  '+' & '.join([dl,drift,
            f'{st:.1f}',fmt(lru),fmt(gpt),fmt(mem),fmt(qdc),f'{delta:+.1f}'])+r' \\')
    lines.append(r'  \midrule')
lines.pop()
lines+=[r'\bottomrule',r'\end{tabular}',r'\end{table*}']
tex=FIGS/'latex_table_v3.tex'; tex.write_text('\n'.join(lines)+'\n'); print(f'  Saved {tex}')

# ── Console summary ────────────────────────────────────────────────────
print()
print('='*84)
print(f'{"Dataset":26s} {"Drift":8s} | {"LRU":6s} {"GPT":6s} {"MEM":6s} {"QDC":6s} | {"Δ":>8s}  {"wins"}')
print('-'*84)
for ds,dl in DS_TITLES_MO1.items():
    for drift,rd in [('sudden',mo1s),('gradual',mo1g)]:
        lru=h2v(rd,ds,'LRU'); gpt=h2v(rd,ds,'GPTCacheStyle')
        mem=h2v(rd,ds,'MemGPTStyle'); qdc=h2v(rd,ds,'DRIP-Dense')
        if None in (lru,gpt,mem,qdc): continue
        d=qdc-max(lru,gpt,mem); w='✓' if d>=0 else '✗'
        print(f'{dl:26s} {drift:8s} | {lru:5.1f} {gpt:6.1f} {mem:6.1f} {qdc:6.1f} | {d:+7.1f}pp {w}')
print('='*84)
print()
print('MO2:')
for ds,dl in DS_TITLES_MO2.items():
    for drift,rd in [('sudden',mo2s),('gradual',mo2g)]:
        lru=h2v(rd,ds,'LRU'); gpt=h2v(rd,ds,'GPTCacheStyle')
        mem=h2v(rd,ds,'MemGPTStyle'); qdc=h2v(rd,ds,'DRIP-Dense')
        if None in (lru,gpt,mem,qdc): continue
        d=qdc-max(lru,gpt,mem); w='✓' if d>=0 else '✗'
        print(f'{dl:26s} {drift:8s} | {lru:5.1f} {gpt:6.1f} {mem:6.1f} {qdc:6.1f} | {d:+7.1f}pp {w}')
print(f'\nAll figures → {FIGS}')

# ───────────────────────────────────────────────────────────────────────
# System-level: 3-bar figure  (Quality | Latency | Overhead)
# Reviewer Concern 4: must show Latency, Overhead, Recall
# ─────────────────────────────────────────────────────────────────────
print('Building system-metrics 3-bar figure...')

MARKERS = {s: m for s, m in zip(
    ['Static','DocArrival','KnowledgeEdit','LRU','GPTCacheStyle',
     'MemGPTStyle','DRIP-Dense','DRIP-Dense-Loose','Oracle'],
    ['s','D','P','o','^','v','*','h','X'])}

SYS_ORDER = ['Static','DocArrival','KnowledgeEdit','LRU',
             'GPTCacheStyle','MemGPTStyle','OnDemandFetch','DRIP-Dense']

def build_system_bars(data_dict, ds_map, title, fname):
    recall, latency, overhead = {}, {}, {}
    for s in SYS_ORDER:
        recall[s] = []; latency[s] = []; overhead[s] = []
    for ds in ds_map:
        if ds not in data_dict: continue
        sm = data_dict[ds]['summary']
        nw = data_dict[ds]['config']['n_windows']
        for s in SYS_ORDER:
            if s not in sm: continue
            m = sm[s]
            r = m.get('recall@5_h2')
            # serve_latency = production-equivalent online cost:
            #   hot-tier KB search (cache/Static/QD/Oracle)
            #   OR hot + remote cold-tier RTT (OnDemand)
            lat = serve_latency_ms(m)
            uc  = m.get('update_latency_mean_ms', m.get('step_latency_mean_ms'))
            if uc is None or r is None: continue
            recall[s].append(float(r))
            latency[s].append(float(lat))
            overhead[s].append(float(uc))
    x = np.arange(len(SYS_ORDER))
    short_lbl = ['Static','Doc\nArrival','Knowl.\nEdit','LRU',
                 'GPTCache\n(fair)','MemGPT\n(fair)','OnDemand','NQM\n(ours)']
    bcolors = [BAR_COLORS.get(s,'#888') for s in SYS_ORDER]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    specs = [
        (recall,   'Post-drift Recall@5 H2 (%)',      '(a) Retrieval Quality'),
        (latency,  'Online serving (ms/query)', '(b) Online Serving Latency'),
        (overhead, 'Offline update (ms/window)', '(c) Offline Update Overhead'),
    ]
    for ax, (vals_d, ylabel, sub_t) in zip(axes, specs):
        means = [float(np.mean(vals_d[s])) if vals_d[s] else 0.0
                 for s in SYS_ORDER]
        bars = ax.bar(x, means, color=bcolors, zorder=3,
                      edgecolor='white', linewidth=0.6)
        qd_i = SYS_ORDER.index('DRIP-Dense')
        bars[qd_i].set_edgecolor('#0D47A1')
        bars[qd_i].set_linewidth(2.2)
        mx = max(means) if max(means) > 0 else 1.0
        for i, (bar, v) in enumerate(zip(bars, means)):
            fmt_v = f'{v:.1f}' if v < 1000 else f'{v:.0f}'
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + mx*0.015, fmt_v,
                    ha='center', va='bottom',
                    fontsize=8 if i != qd_i else 8.5,
                    fontweight='bold' if i == qd_i else 'normal',
                    color='#1565C0' if i == qd_i else '#333')
        ax.set_xticks(x)
        ax.set_xticklabels(short_lbl, fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(sub_t, fontsize=11, fontweight='bold', pad=6)
        ax.grid(True, axis='y', alpha=0.22, ls=':', zorder=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    scope = 'on ' + next(iter(ds_map.values())) if len(ds_map) == 1 else \
            'averaged over ' + str(len(ds_map)) + ' datasets'
    fig.suptitle(title + '\n(' + ENC_DENSE + '  ' + scope + ')',
                 fontsize=11.5, fontweight='bold', y=1.01)
    fig.tight_layout()
    return fig

DS_TITLES_SYS = {'hotpotqa_comparison': 'HotpotQA-comp'}
fig = build_system_bars(mo1g, DS_TITLES_SYS,
    'Hot-tier cache quality vs. production-equivalent cost (HotpotQA-comp, pool=200k / KB=21k, gradual drift)',
    'system_bars_mo1_sudden')
save(fig, 'system_bars_mo1_sudden')

fig = build_system_bars(mo1g, DS_TITLES_SYS,
    'Hot-tier cache quality vs. production-equivalent cost (HotpotQA-comp, pool=200k / KB=21k, gradual drift)',
    'system_bars_mo1_gradual')
save(fig, 'system_bars_mo1_gradual')


# Aggregate for LaTeX table
agg_mo1_s = {s: {'serve_latency': [], 'update_latency': [], 'h2': []}
             for s in SYS_ORDER + ['Oracle']}
for ds in DS_TITLES_SYS:
    if ds not in mo1g: continue
    sm = mo1g[ds]['summary']
    for s in agg_mo1_s:
        if s not in sm: continue
        m = sm[s]
        qlat = serve_latency_ms(m)  # production-equivalent online serving
        ulat = m.get('update_latency_mean_ms', m.get('step_latency_mean_ms'))
        r = m.get('recall@5_h2')
        if ulat is None or r is None: continue
        agg_mo1_s[s]['serve_latency'].append(float(qlat))
        agg_mo1_s[s]['update_latency'].append(float(ulat))
        agg_mo1_s[s]['h2'].append(float(r))

# ── System metrics LaTeX table ─────────────────────────────────────────
print('Building system-metrics LaTeX table…')
lines = [
    r'\begin{table*}[t]', r'\centering', r'\small',
    r'\caption{Hot-tier cache cost-quality trade-off under gradual query-drift '
    r'(single-hop HotpotQA-comp; pool=200{,}000 cold-tier docs, KB=21{,}350 hot-tier slots, ratio $\approx$\,10$\times$). '
    r'\textsc{Online Serving} = production-equivalent per-query latency. Cache strategies, '
    r'Static, QD, and Oracle pay only the in-memory hot-tier FAISS search ($\approx$1.6\,ms/query '
    r'over 21K docs). OnDemandFetch additionally pays a remote cold-tier ANN search per query, '
    r'modelled as $T_{\text{remote}}=5\,T_{\text{local}}$ (mid-point of the 3--8$\times$ '
    r'in-mem-vs-remote gap reported by VectorDBBench~2024 over 100K--1M-doc indexes; network '
    r'RTT is already absorbed in those end-to-end p50 numbers). '
    r'\textsc{Update Overhead} = measured offline background hot-tier rewrite latency per window. '
    r'H2\,R@5 = post-drift Recall@5 (higher is better).}',
    r'\label{tab:system_metrics}',
    r'\setlength{\tabcolsep}{6pt}',
    r'\begin{tabular}{lrrrl}',
    r'\toprule',
    r'Strategy & Online Serving (ms/query) & Update Overhead (ms/window) & H2 R@5 (\%) & Family \\',
    r'\midrule',
]
FAMILY = {
    'Static': 'no-update', 'DocArrival': 'supply-driven',
    'KnowledgeEdit': 'edit-based', 'LRU': 'cache (recency)',
    'GPTCacheStyle': 'cache (semantic)', 'MemGPTStyle': 'cache (importance)',
    'OnDemandFetch': 'online fetch',
    'DRIP-Dense': r'\textbf{ours}', 'Oracle': 'upper-bound',
}
order = ['Static','DocArrival','KnowledgeEdit','LRU','GPTCacheStyle',
         'MemGPTStyle','OnDemandFetch','DRIP-Dense','Oracle']
for sn in order:
    d = agg_mo1_s.get(sn, {})
    if not d.get('serve_latency'): continue
    qlat = float(np.mean(d['serve_latency']))
    ulat = float(np.mean(d['update_latency']))
    h2 = float(np.mean(d['h2']))
    fmt_name = r'\textbf{NQM (ours)}' if sn == 'DRIP-Dense' \
               else BAR_LABELS.get(sn, sn)
    lines.append(f'  {fmt_name} & {qlat:6.1f} & {ulat:6.1f} & {h2:5.1f} & {FAMILY[sn]} \\\\')
lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
sys_tex = FIGS / 'system_metrics_table.tex'
sys_tex.write_text('\n'.join(lines) + '\n')
print(f'  Saved {sys_tex}')

# ── Console: cost-quality leaderboard ─────────────────────────────────
print()
print('System-level summary  (Mo1 HotpotQA-comp gradual):')
print(f'{"Strategy":22s} {"Serve(ms/q)":>15s} {"Update(ms/win)":>15s} {"H2 R@5%":>9s}')
print('-' * 70)
for sn in order:
    d = agg_mo1_s.get(sn, {})
    if not d.get('serve_latency'): continue
    print(f'{sn:22s} {np.mean(d["serve_latency"]):15.1f} {np.mean(d["update_latency"]):15.1f} {np.mean(d["h2"]):9.1f}')

# =============================================================================
# v3.1 Reviewer-response additions
# 1. TREC-COVID temporal coverage (real gradual drift validation)
# 2. Mo2 dense-vs-graph ablation bar
# 3. Combined Mo1+Mo2 intro figure (QD works single-hop, fails multi-hop)
# 4. Reorganise figures_v3/ into paper subdirs
# =============================================================================
import shutil

# ── 1. StreamingQA temporal recall@5 ─────────────────────────────────────────
# Liška et al., StreamingQA (ICML 2022): real 14-yr news QA stream from
# FT/WSJ/etc.  We bin the 4.8K dated test queries into 5 year-rounds and
# track per-window Recall@5 against the round-specific gold articles.
print('\nBuilding StreamingQA temporal recall@5 figure…')
TREC_TEMPORAL = ROOT.parent / 'experiments/direct/data/results_streamingqa_temporal.json'
if TREC_TEMPORAL.exists():
    tc = load(TREC_TEMPORAL)
    ds_key = list(tc.keys())[0]
    tc_sm = tc[ds_key]['summary']
    nw_tc = len(tc_sm['Static']['recall@5_per_window'])
    x_tc = np.arange(1, nw_tc + 1)
    # Round boundaries baked in by stream sampler (head_rounds=[1,2] over
    # 25 windows; tail_rounds=[3,4,5] over 25 windows).
    ROUNDS = [(1, 13,  'R1 (2008-10)\nFin.crisis',  '#EDF4FC'),
              (14, 25, 'R2 (2011-13)\nArab Spring', '#E8EFFA'),
              (26, 33, 'R3 (2014-16)\nBrexit',      '#FDF5E6'),
              (34, 42, 'R4 (2017-18)\nTrump era',   '#F5EAF5'),
              (43, nw_tc,'R5 (2019-20)\nCOVID',     '#EAF5EA')]

    fig_tc, ax_tc = plt.subplots(figsize=(11, 4.5))
    for (rs, re, rlbl, rbg) in ROUNDS:
        ax_tc.axvspan(rs - 0.5, re + 0.5, facecolor=rbg, alpha=1.0, zorder=0)
        ax_tc.text((rs + re) / 2, 96, rlbl, ha='center', va='bottom',
                   fontsize=7.8, color='#444', fontweight='bold')
    for bound in [13.5, 25.5, 33.5, 42.5]:
        ax_tc.axvline(bound, color='#888', lw=1.0, ls='--', alpha=0.6, zorder=4)

    # Oracle
    oc = np.array(tc_sm['Oracle']['recall@5_per_window'], dtype=float)
    ax_tc.plot(x_tc, oc, color=PALETTE['Oracle'][0], lw=2.4, ls='-',
               alpha=0.9, label='Oracle (upper bound)', zorder=6)
    # Static
    sc = np.array(tc_sm['Static']['recall@5_per_window'], dtype=float)
    ax_tc.plot(x_tc, sc, color=PALETTE['Static'][0], lw=2.0,
               ls=(0, (2, 2)), alpha=0.85, label='Static (frozen KB)', zorder=5)
    # Strong baselines plus adaptive routes.
    for sn_tc in ['LRU', 'GPTCacheStyle', 'MemGPTStyle', 'TemporalAware', 'OnDemandFetch', 'DRIP-Dense']:
        cpw = tc_sm.get(sn_tc, {}).get('recall@5_per_window')
        if not cpw: continue
        col_tc, ls_tc = PALETTE.get(sn_tc, ('#888', '--'))
        _hot = ('DRIP-Dense', 'OnDemandFetch', 'TemporalAware')
        lw_tc = 2.2 if sn_tc in _hot else 1.5
        alpha_tc = 0.9 if sn_tc in _hot else 0.6
        z_tc = 5 if sn_tc in _hot else 3
        ax_tc.plot(x_tc, np.array(cpw, dtype=float), color=col_tc, lw=lw_tc,
                   ls=ls_tc, alpha=alpha_tc, label=LABELS.get(sn_tc, sn_tc), zorder=z_tc)

    ax_tc.set_xlim(0.5, nw_tc + 0.5); ax_tc.set_ylim(0, 105)
    ax_tc.set_xlabel('Window index (time →)', fontsize=11)
    ax_tc.set_ylabel('Recall@5 (%)', fontsize=11)
    _tc_cfg = tc[ds_key]['config']
    ax_tc.set_title(
        f"StreamingQA: 14-year real news QA stream (Liška et al., ICML 2022) — 5 year-rounds\n"
        f"cold pool={_tc_cfg['pool_size']:,} docs · hot KB={_tc_cfg['kb_budget']:,} slots ({_tc_cfg['pool_size']/_tc_cfg['kb_budget']:.0f}× ratio)  |  Encoder: BAAI/bge-base-en-v1.5",
        fontsize=10.5, fontweight='bold')
    ax_tc.legend(loc='lower left', frameon=False, fontsize=9, ncol=4)
    ax_tc.grid(True, axis='y', alpha=0.2, ls=':')
    fig_tc.tight_layout()
    save(fig_tc, 'streamingqa_temporal')
    print('  StreamingQA temporal recall@5 figure saved.')
else:
    print('  StreamingQA temporal file not found, skipping.')

# ── 2. Mo2 dense-vs-graph ablation bar ───────────────────────────────────────
print('\nBuilding Mo2 dense-vs-graph ablation figure…')
ENC_BGE = 'Encoder: BAAI/bge-large-en-v1.5'
DG_FILES = {
    'hotpotqa_bridge': (
        ROOT.parent / 'experiments/hidden/data/results_100w_hotpot_bridge_sudden_dense_bge_v2.json',
        ROOT.parent / 'experiments/hidden/data/results_100w_hotpot_bridge_sudden_graph_bge_v2.json',
        'hotpotqa_expanded'),
    '2wiki': (
        ROOT.parent / 'experiments/hidden/data/results_100w_2wiki_sudden_dense_bge_v2.json',
        ROOT.parent / 'experiments/hidden/data/results_100w_2wiki_sudden_graph_bge_v2.json',
        '2wikimultihopqa'),
    'musique': (
        ROOT.parent / 'experiments/hidden/data/results_100w_musique_sudden_dense_bge_v2.json',
        ROOT.parent / 'experiments/hidden/data/results_100w_musique_sudden_graph_bge_v2.json',
        'musique'),
}
DG_LABELS = {'hotpotqa_bridge': 'HotpotQA-Bridge',
             '2wiki': '2WikiMultihopQA', 'musique': 'MuSiQue'}
DG_STRATS = ['Static', 'DRIP-Dense', 'Oracle']
DC = {'Static': '#9E9E9E',  'DRIP-Dense': '#1565C0',  'Oracle': '#C62828'}
GC = {'Static': '#BDBDBD',  'DRIP-Dense': '#42A5F5',  'Oracle': '#EF9A9A'}

all_dg_ok = all(p[0].exists() and p[1].exists() for p in DG_FILES.values())
if all_dg_ok:
    ds_keys = list(DG_FILES.keys())
    n_ds = len(ds_keys); n_s = len(DG_STRATS)
    group_w = 0.55; bar_w = group_w / 2 * 0.82
    x_dg = np.arange(n_ds)

    fig_dg, ax_dg = plt.subplots(figsize=(10, 5.2))
    for si, sn_dg in enumerate(DG_STRATS):
        dense_v, graph_v = [], []
        for dsk in ds_keys:
            dp, gp, jk = DG_FILES[dsk]
            dd = json.load(open(dp)); gd = json.load(open(gp))
            dense_v.append(dd.get(jk, {}).get('summary', {}).get(sn_dg, {}).get('recall@5_h2', 0))
            graph_v.append(gd.get(jk, {}).get('summary', {}).get(sn_dg, {}).get('recall@5_h2', 0))
        off = (si - 1) * group_w
        b1 = ax_dg.bar(x_dg + off - bar_w / 2, dense_v, bar_w,
                       color=DC[sn_dg], zorder=3,
                       label=f'{BAR_LABELS.get(sn_dg, sn_dg)} (dense)')
        b2 = ax_dg.bar(x_dg + off + bar_w / 2, graph_v, bar_w,
                       color=GC[sn_dg], zorder=3, hatch='//',
                       edgecolor='white', linewidth=0.4,
                       label=f'{BAR_LABELS.get(sn_dg, sn_dg)} (graph-aug)')
        for xi, (dv, gv) in zip(x_dg + off, zip(dense_v, graph_v)):
            delta = gv - dv
            ax_dg.text(xi + bar_w / 2, gv + 0.6, f'{delta:+.1f}',
                       ha='center', va='bottom', fontsize=7.5,
                       color='#B71C1C' if abs(delta) >= 2 else '#555')
    ax_dg.set_xticks(x_dg)
    ax_dg.set_xticklabels([DG_LABELS[d] for d in ds_keys], fontsize=11)
    ax_dg.set_ylabel('Post-drift Recall@5 H2 (%)', fontsize=11)
    ax_dg.set_title(
        f'Mo2 Ablation: Dense vs Graph-augmented Retrieval  ({ENC_BGE} · Sudden drift)\n'
        'Δ annotations = graph − dense.  Graph retrieval does not rescue QD: KB content is the bottleneck.',
        fontsize=10.5, fontweight='bold')
    ax_dg.legend(frameon=False, fontsize=9, ncol=3, loc='upper right')
    ax_dg.grid(True, axis='y', alpha=0.2, ls=':')
    fig_dg.tight_layout()
    save(fig_dg, 'mo2_dense_vs_graph')
    print('  Mo2 dense-vs-graph figure saved.')
else:
    print('  Mo2 graph/dense files missing, skipping.')

# ── 3. Intro diagnostic figure: 3 panels (Mo1-cov | Mo1-rec | Mo2-rec) ──────
print('\nBuilding intro diagnostic figure (3-panel)…')

def draw_intro_panel3(ax, res, mk, ylabel, ds_title, show_on_demand=False):
    """Clean single panel for the intro figure."""
    INTRO_SHOW = ['Static', 'LRU', 'GPTCacheStyle', 'MemGPTStyle',
                  'KnowledgeEdit', 'DRIP-Dense', 'Oracle']
    if show_on_demand:
        INTRO_SHOW = ['Static', 'LRU', 'GPTCacheStyle', 'MemGPTStyle',
                      'KnowledgeEdit', 'LogDrivenArrival', 'OnDemandFetch',
                      'DRIP-Dense', 'Oracle']
    cfg = res['config']; nw = cfg['n_windows']; half = nw // 2
    x = np.arange(1, nw + 1)
    ax.axvspan(0.5, half+0.5, facecolor=PRE_BG, alpha=1.0, zorder=0)
    ax.axvspan(half+0.5, nw+0.5, facecolor=POST_BG, alpha=1.0, zorder=0)
    for name in INTRO_SHOW:
        if name not in res['summary']: continue
        per = res['summary'][name].get(mk)
        if not per: continue
        col, ls = PALETTE.get(name, ('#888', '-'))
        vals = smooth(np.asarray(per, dtype=float), half, w=5)
        lw = 3.0 if name == 'DRIP-Dense' else (2.2 if name == 'Oracle' else
              (2.6 if name == 'OnDemandFetch' else 1.8))
        al = 1.0 if name in ('DRIP-Dense', 'OnDemandFetch') else (0.9 if name == 'Oracle' else 0.72)
        z  = 7 if name == 'DRIP-Dense' else (8 if name == 'OnDemandFetch' else (6 if name == 'Oracle' else 3))
        ax.plot(x, vals, color=col, linestyle=ls, linewidth=lw, alpha=al,
                zorder=z, label=LABELS.get(name, name),
                solid_capstyle='round', dash_capstyle='round')
    ax.axvline(half+0.5, color='#333', ls='-', lw=1.4, alpha=0.6, zorder=9)
    ax.text(half+0.5, 104, 'drift', ha='center', va='bottom', fontsize=8.5,
            fontweight='bold', color='#333',
            bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='#333', lw=0.9, alpha=0.9))
    ax.set_xlim(0.5, nw+0.5); ax.set_ylim(0, 110)
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_xlabel('Window index (time →)', fontsize=10)
    ax.set_title(ds_title, fontsize=11, fontweight='bold', pad=6)
    ax.grid(True, axis='y', alpha=0.2, ls=':', zorder=1)

def draw_intro_system_bars(ax, res, n_queries, panel_title, font_scale=1.0, bar_positions=None):
    """Panel (c): online serving latency (hot-tier KB search ± cold-pool search) and
    offline background update overhead + Recall@5 H2.

    Cost taxonomy:
      serve_latency  = query_latency_mean_ms  = total online serving latency:
                       hot-tier KB search for ALL strategies (~79-81 ms);
                       OnDemandFetch additionally pays cold-pool search → ~136 ms.
      update_latency = update_latency_mean_ms = OFFLINE background KB rewrite per window
                       (cache/QD pay this; OnDemand = 0 ms).
      Result: cache strategies serve fast but update is visible; OnDemand has
      higher serving cost with zero update overhead.
    """
    methods = ['LRU', 'GPTCacheStyle', 'MemGPTStyle', 'KnowledgeEdit',
               'OnDemandFetch', 'DRIP-Dense']
    short = {
        'LRU': 'LRU', 'GPTCacheStyle': 'GPTCache', 'MemGPTStyle': 'MemGPT',
        'KnowledgeEdit': 'RECIPE', 'OnDemandFetch': 'OnDemand',
        'DRIP-Dense': 'QD',
    }
    vals = {m: {'recall': 0.0, 'serve_latency': 0.0, 'update_latency': 0.0}
            for m in methods}
    for name in methods:
        st = res['summary'].get(name)
        if not st:
            continue
        vals[name]['recall'] = float(st.get('recall@5_h2', 0.0))
        # serve_latency = production-equivalent online serving:
        #   cache / Static / QD: hot-tier KB search only
        #   OnDemand: hot-tier check + remote cold-tier search (RTT + scaled)
        vals[name]['serve_latency'] = serve_latency_ms(st)
        # update_latency = background KB rewrite/evict per window (offline)
        vals[name]['update_latency'] = float(
            st.get('update_latency_mean_ms', st.get('step_latency_mean_ms', 0.0))
        )

    # unit is embedded in the title; NO ylabel to avoid axis-label/tick overlap
    metrics = [
        ('Recall@5 H2 (%) ↑',
         'recall', False),
        ('Online serving (ms/query) ↓',
         'serve_latency', False),
        ('Offline update (ms/window) ↓',
         'update_latency', True),
    ]
    ax.axis('off')
    ax.set_title(panel_title, fontsize=9.8*font_scale, fontweight='bold', pad=4)
    # Keep mini-panels narrow but readable; only key bars are annotated below.
    _default_pos = [(0.02, 0.25, 0.30, 0.62),
                    (0.35, 0.25, 0.30, 0.62),
                    (0.68, 0.25, 0.30, 0.62)]
    positions = bar_positions if bar_positions is not None else _default_pos
    for (title, key, use_log), pos in zip(metrics, positions):
        sub = ax.inset_axes(pos)
        y = [vals[m][key] for m in methods]
        x = np.arange(len(methods))
        colors = [PALETTE.get(m, ('#888', '-'))[0] for m in methods]
        edges = ['#111' if m in ('DRIP-Dense', 'OnDemandFetch') else 'none'
                 for m in methods]
        bars = sub.bar(x, y, color=colors, edgecolor=edges, linewidth=1.0,
                       alpha=0.92, zorder=3)
        if use_log:
            sub.set_yscale('symlog', linthresh=0.1)
        ymax = max(y) if max(y) > 0 else 1.0
        # Reserve 30% headroom so value labels never collide with subplot title.
        if not use_log:
            sub.set_ylim(0, ymax * 1.30)
        for m, bar, val in zip(methods, bars, y):
            # Annotate only the methods that matter for the story. Dense labels
            # in the compressed intro panel collide with titles and ticks.
            if m not in ('DRIP-Dense', 'OnDemandFetch') and abs(val - ymax) > 1e-6:
                continue
            if val <= 0 and key != 'update_latency':
                continue
            txt = f'{val:.0f}' if key == 'recall' else (
                f'{val:.0f}' if val >= 100 else f'{val:.1f}')
            is_hi = (m in ('DRIP-Dense', 'OnDemandFetch')
                     or abs(val - ymax) < 1e-6)
            sub.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + ymax * 0.02,
                     txt, ha='center', va='bottom',
                     fontsize=6.6*font_scale,
                     fontweight='bold' if is_hi else 'normal',
                     color='#1565C0' if m == 'DRIP-Dense' else (
                         '#00838F' if m == 'OnDemandFetch' else '#333'))
        sub.set_title(title.replace('Online serving', 'Serve').replace('Offline update', 'Update'),
                      fontsize=7.4*font_scale, fontweight='bold', pad=2)
        sub.set_xticks(x)
        sub.set_xticklabels([short[m] for m in methods], rotation=55,
                            ha='right', fontsize=6.2*font_scale)
        sub.tick_params(axis='y', labelsize=6.8*font_scale, pad=1)
        sub.grid(True, axis='y', alpha=0.22, ls=':', zorder=0)
        sub.set_ylabel('')


mo1_case = mo1g.get('hotpotqa_comparison')
mo2_case = mo2g.get('2wikimultihopqa')

if mo1_case and mo2_case:
    # 2-row layout: (a)+(b) top, (c) bottom full-width
    # Aspect ratio ≈ 1.6:1  (was 4.2:1) — much more readable in double-column figure*
    fig_i = plt.figure(figsize=(13.5, 8.6))
    _gs_i = fig_i.add_gridspec(2, 2, height_ratios=[1.3, 0.95],
                               hspace=0.50, wspace=0.30)
    axes_i = [fig_i.add_subplot(_gs_i[0, 0]),   # (a) single-hop
              fig_i.add_subplot(_gs_i[0, 1]),   # (b) multi-hop
              fig_i.add_subplot(_gs_i[1, :])]   # (c) full-width cost bars
    mc1 = mo1_case['config']; mc2 = mo2_case['config']

    # Panel (a): single-hop recall, with OnDemand
    draw_intro_panel3(axes_i[0], mo1_case, 'recall@5_per_window', 'Recall@5 (%)',
        f'(a) Single-hop Recall@5\nHotpotQA-comp  ·  pool={mc1["pool_size"]:,}  KB={mc1["kb_budget"]:,}',
        show_on_demand=True)
    # Panel (b): multi-hop recall, with OnDemand
    draw_intro_panel3(axes_i[1], mo2_case, 'recall@5_per_window', 'Recall@5 (%)',
        f'(b) Multi-hop Recall@5\n2WikiMultihopQA  ·  pool={mc2["pool_size"]:,}  KB={mc2["kb_budget"]:,}',
        show_on_demand=True)
    # Panel (c): system metrics on the single-hop representative task only
    nq1 = mc1['n_windows'] * mc1['window_size']
    draw_intro_system_bars(
        axes_i[2], mo1_case, nq1,
        '(c) Hot-tier cost/quality trade-off\nHotpotQA-comp · pool=200k / KB=21k',
        font_scale=1.22,
        bar_positions=[(0.03, 0.12, 0.29, 0.80),
                       (0.36, 0.12, 0.29, 0.80),
                       (0.69, 0.12, 0.29, 0.80)],
    )

    # Non-obstructing callouts: keep away from the central drift labels.
    axes_i[0].text(0.03, 0.06, 'QD improves persistent-KB recall',
                   transform=axes_i[0].transAxes, ha='left', va='bottom',
                   fontsize=8.4, fontweight='bold', color='#1565C0',
                   bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#1565C0',
                             lw=1.0, alpha=0.92))
    axes_i[1].text(0.03, 0.06, 'Multi-hop gap remains',
                   transform=axes_i[1].transAxes, ha='left', va='bottom',
                   fontsize=8.4, fontweight='bold', color='#C62828',
                   bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#C62828',
                             lw=1.0, alpha=0.92))

    # Shared legend from panels (a,b); panel (c) has compact method labels.
    handles, labels = [], []
    for ax in axes_i[:2]:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi); labels.append(li)
    fig_i.legend(handles, labels, loc='lower center', ncol=5, frameon=False,
                 bbox_to_anchor=(0.5, 0.005), fontsize=9.5,
                 handlelength=3.0, columnspacing=1.2)
    fig_i.suptitle(
        'Hot-tier cache admission under query drift  ·  no document timestamps  ·  QD is a query-side baseline',
        fontsize=11, fontweight='bold', y=0.998)
    fig_i.tight_layout(rect=[0, 0.065, 1, 0.972])
    save(fig_i, 'intro_combined')
    print('  4-panel intro figure saved.')
else:
    print('  Missing data for intro_combined.')

# ── 4. Reorganise figures_v3/ into paper subdirs ─────────────────────────────
print('\nOrganising figures into paper subdirs…')
INTRO2   = FIGS / 'intro'
EXPS2    = FIGS / 'experiments'
APP2     = FIGS / 'appendix'
for _d in (INTRO2, EXPS2, APP2):
    _d.mkdir(exist_ok=True)
    # Keep paper_figs deterministic when figure numbering changes.
    for _old in list(_d.glob('*.pdf')) + list(_d.glob('*.png')) + list(_d.glob('*.tex')):
        _old.unlink()

def mv(src_stem, dst_dir, new_stem=None):
    ns = new_stem or src_stem
    for ext in ('pdf', 'png'):
        s = FIGS / f'{src_stem}.{ext}'
        d = dst_dir / f'{ns}.{ext}'
        if s.exists(): shutil.copy2(s, d)

# Intro figures: Fig. 0 = workload evidence, Fig. 1 = real stream, Fig. 2 = QD diagnostic.
_mo0 = ROOT / 'motivation_0/figures/user_query_topic_drift'
for _ext in ('pdf', 'png'):
    _src = _mo0.with_suffix(f'.{_ext}')
    _dst = INTRO2 / f'fig0_intro_user_query_topic_drift.{_ext}'
    if _src.exists(): shutil.copy2(_src, _dst)
mv('streamingqa_temporal', INTRO2, 'fig1_intro_streamingqa_static_cache')
mv('intro_combined',       INTRO2, 'fig2_intro_qd_diagnostic')
# Experiments (used by the main experiments section)
mv('mo1_curves_gradual',  EXPS2,  'fig2a_mo1_curves_gradual')
mv('mo2_curves_gradual',  EXPS2,  'fig2b_mo2_curves_gradual')
mv('h2_bar_gradual',      EXPS2,  'fig3_h2_bar_gradual')
mv('system_bars_mo1_sudden', EXPS2, 'fig4_system_metrics')
mv('streamingqa_temporal', EXPS2,  'fig5_real_temporal_streamingqa')
# Appendix
mv('mo1_curves_sudden',   APP2,   'figA1_mo1_curves_sudden')
mv('mo2_curves_sudden',   APP2,   'figA2_mo2_curves_sudden')
mv('h2_bar_sudden',       APP2,   'figA3_h2_bar_sudden')
mv('mo2_dense_vs_graph',  APP2,   'figA4_mo2_dense_vs_graph')
# Tables
for _src, _dst, _nm in [
    ('latex_table_v3.tex',       EXPS2, 'table1_main_results.tex'),
    ('system_metrics_table.tex', EXPS2, 'table2_system_metrics.tex'),
]:
    _s = FIGS / _src; _d2 = _dst / _nm
    if _s.exists(): shutil.copy2(_s, _d2)

# Remove root-level loose files (keep only organised subdirs)
import os
for _f in FIGS.iterdir():
    if _f.is_file() and _f.suffix in ('.pdf', '.png', '.tex'):
        _f.unlink()
print('  Cleaned root-level intermediate files.')

print('\nFinal figure layout:')
for _sub in (INTRO2, EXPS2, APP2):
    _fl = sorted(f.name for f in _sub.iterdir()
                 if f.suffix in ('.pdf', '.png', '.tex'))
    print(f'  {_sub.name}/'); [print(f'    {f}') for f in _fl]
print('\nAll done — plot_v3.py v3.1')
