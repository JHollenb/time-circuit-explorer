"""Render published scalar evidence only; performs no model computation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from encoder_atoms import read,save,sha


def render(source,out):
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True);r=read(source/'report.json');c=read(source/'compile-receipt.json');methods=['00','stream','quadratic'];colors=['#9ba6b8','#6c59c7','#2b9b8f'];names=['Centroid','Rank-one state','Full-direction quadratic'];regions=['all','active_both','padding_both']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#374151','text.color':'#202839','axes.titleweight':'bold','svg.hashsalt':'tce-rank-one-tail-v1'})
    fig,axes=plt.subplots(1,3,figsize=(14,4),gridspec_kw={'width_ratios':[1.15,1.25,.85]});fig.patch.set_facecolor('#fbfcfe')
    ax=axes[0];x=np.arange(3);width=.23
    for j,(m,label,color) in enumerate(zip(methods,names,colors)):
        values=[100*next(z for z in r['pairedScores'] if z['method']==m and z['stage']=='state' and z['region']==region)['attentionEffect']['relativeRms'] for region in regions]
        bars=ax.bar(x+(j-1)*width,values,width,color=color,label=label)
        for bar,value in zip(bars,values):ax.text(bar.get_x()+width/2,value+.7,f'{value:.1f}',ha='center',fontsize=8)
    ax.set_xticks(x,['All rows','Active','Padding']);ax.set_ylabel('Paired state error / attention effect (%)');ax.set_ylim(0,65);ax.set_title('A smaller score loses much of the gain',fontsize=11);ax.legend(frameon=False,fontsize=8,loc='upper left');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax=axes[1]
    def heads(method,region):return {z['queryHead']:z['score']['relativeRms'] for z in r['headScores'] if z.get('kind')=='paired' and z['method']==method and z['region']==region and z['precision']=='rounded' and z['queryHead'] is not None}
    h=heads('stream','active_both');base=heads('00','active_both');delta=[100*(h[i]-base[i]) for i in range(32)];ax.bar(np.arange(32),delta,color=['#d37b72' if a>0 else '#6c59c7' for a in delta]);ax.axhline(0,color='#64748b',lw=.7);ax.set_xlabel('Query head');ax.set_ylabel('Active head error change (percentage points)');ax.set_title('16 active heads improve; 16 worsen',fontsize=11);ax.set_xticks([0,8,16,24,31]);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax=axes[2];values=[100*g['resolvedCapturedScoreEnergy']/g['sourceProductScoreEnergy'] for g in c['groups']];ax.barh(np.arange(8),values,color='#8997b1');ax.set_yticks(np.arange(8),[f'Group {i}' for i in range(8)]);ax.invert_yaxis();ax.set_xlim(0,21);ax.set_xlabel('Source score energy retained (%)');ax.set_title('One direction retains 8–15%',fontsize=11)
    for i,value in enumerate(values):ax.text(value+.3,i,f'{value:.1f}',va='center',fontsize=9)
    fig.suptitle('Rank-one streaming state: arithmetic closes; representation remains limited',x=.035,ha='left',fontsize=15,weight='bold');fig.text(.035,.025,'One retained prompt pair, first encoder block. No fallbacks; direct and streaming BF16 banks identical. Full KV selection and original receiving equations remain paid.',fontsize=9,color='#64748b');fig.tight_layout(rect=[.02,.09,.99,.9]);fig.savefig(out/'response-comparison.svg',metadata={'Date':None});fig.savefig(out/'response-comparison.png',dpi=150);plt.close(fig)
    path=out/'response-comparison.svg';path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    save(out/'figure.json',{'schema':'tce-rank-one-tail-figure-v1','sourceReportSHA256':sha(source/'report.json'),'sourceCompileReceiptSHA256':sha(source/'compile-receipt.json'),'rendererSHA256':sha(__file__),'matplotlib':matplotlib.__version__,
        'files':{p.name:sha(p) for p in [out/'response-comparison.svg',out/'response-comparison.png']},'scope':'Local rendering of published scalar measurements only, not another scientific run. Dependent rows/heads, no confidence interval or generalization claim.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('out');a=p.parse_args();render(a.source,a.out)
