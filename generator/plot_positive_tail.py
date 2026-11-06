"""Plot already published scalar records; no model or tensor-state execution."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def plot(source,out,preview):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source=Path(source);out=Path(out);preview=Path(preview);r=json.loads((source/'report.json').read_text());diag=json.loads(gzip.decompress((source/'tail-diagnostics.json.gz').read_bytes()))['rows']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#485164','text.color':'#273244','axes.edgecolor':'#c9d0db','svg.fonttype':'none'})
    fig,ax=plt.subplots(1,3,figsize=(12.4,3.9),gridspec_kw={'width_ratios':[1,1.5,1]},layout='constrained');fig.set_facecolor('#fbfcfe')
    gray='#9da8b8';blue='#5374b8';green='#23918a';black='#343d4b';regions=['all','active_both','padding_both']
    for method,offset,color,label in [('00',-.18,gray,'Centroid'),('quadratic',.18,blue,'Positive quadratic')]:
        values=[100*next(x['attentionEffect']['relativeRms'] for x in r['pairedScores'] if x['method']==method and x['stage']=='state' and x['region']==region) for region in regions]
        bars=ax[0].bar([i+offset for i in range(3)],values,.34,color=color,label=label)
        ax[0].bar_label(bars,fmt='%.1f',padding=3,fontsize=8)
    ax[0].set_xticks(range(3),['All','Active','Padded']);ax[0].set_ylim(0,52);ax[0].set_ylabel('Paired state error / exact attention effect (%)');ax[0].set_title('The receiving state improves',loc='left',pad=15);ax[0].legend(frameon=False,fontsize=8)
    for region,offset,color,label in [('active_both',-.12,blue,'Active'),('padding_both',.12,green,'Padded')]:
        rows={m:{x['queryHead']:x['score']['relativeRms'] for x in r['headScores'] if x['method']==m and x['kind']=='paired' and x['precision']=='rounded' and x['region']==region and x['queryHead'] is not None} for m in ['00','quadratic']}
        points=[(h,rows['quadratic'][h]/v) for h,v in rows['00'].items() if v is not None and v>0 and rows['quadratic'][h] is not None]
        ax[1].scatter([h+offset for h,v in points],[v for h,v in points],s=17,c=color,label=label,alpha=.85)
    ax[1].axhline(1,color=black,lw=.8,ls='--');ax[1].set_xlim(-1,32);ax[1].set_ylim(0,1.9);ax[1].set_xticks([0,8,16,24,31]);ax[1].set_xlabel('Query head');ax[1].set_ylabel('Quadratic / centroid head-change error');ax[1].set_title('Some head responses still worsen',loc='left',pad=15);ax[1].legend(frameon=False,fontsize=8);ax[1].text(.02,.03,'Below 1: smaller error. Undefined ratios omitted.',transform=ax[1].transAxes,fontsize=7,color='#6c7686')
    fields=['anchorMassBelowMinus1','quadraticMassBelowMinus1','exactMassBelowMinus1'];values=[100*sum(x['mean'] for x in diag if x['field']==field and x['region']=='padding' and x['queryHead'] is None)/2 for field in fields]
    bars=ax[2].bar(['Anchor','Quadratic','Exact'],values,color=[gray,blue,black],width=.65);ax[2].bar_label(bars,fmt='%.2f',padding=3,fontsize=8);ax[2].set_ylim(0,22);ax[2].set_ylabel('Mean tail probability on t < −1 (%)');ax[2].set_title('Negative-tail overweight remains',loc='left',pad=15)
    fig.suptitle('One retained prompt pair · fixed support and routes · original receiving consumer offline',fontsize=11,fontweight='medium')
    out.parent.mkdir(parents=True,exist_ok=True);preview.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,facecolor=fig.get_facecolor());fig.savefig(preview,dpi=160,facecolor=fig.get_facecolor());plt.close(fig)
    # Matplotlib path lines carry redundant trailing spaces; keep generated Git text clean.
    out.write_text('\n'.join(line.rstrip() for line in out.read_text().splitlines())+'\n')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    receipt={'schema':'tce-positive-tail-derived-figure-v1','sourceReportSHA256':sha(source/'report.json'),'sourceDiagnosticsGzipSHA256':sha(source/'tail-diagnostics.json.gz'),'svgSHA256':sha(out),'pngSHA256':sha(preview),'scope':'Local rendering of already published scalar records, no new model computation. Per-head ratios retain countertrends; no epsilon for undefined denominators. Tail bars average equal-size padded cohorts from two dependent prompts, not independent evidence or causal error attribution.'}
    (out.parent/'figure.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('svg');p.add_argument('png');a=p.parse_args();plot(a.source,a.svg,a.png)
