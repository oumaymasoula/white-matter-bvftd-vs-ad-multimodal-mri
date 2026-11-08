"""Reproduce descriptive parcel checks and complete targeted sensitivity tables.

Usage: python parcel_and_sensitivity_review.py --regional-review FILE
       --sensitivity-review FILE --out DIRECTORY
No models are refitted. Source CIs and q-values are preserved.
"""
from pathlib import Path
import argparse, io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--regional-review',required=True)
    p.add_argument('--sensitivity-review',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    text=Path(args.regional_review).read_text()
    raw=text.split('FreeSurfer_DK68_cortical_thickness_results.tsv =====')[1].split('=====')[0].strip()
    d=pd.read_csv(io.StringIO(raw),sep='\t')
    n=d[d.dataset.eq('NIFD')].set_index('parcel')
    a=d[d.dataset.eq('ADNI')&d.comparison.eq('Alzheimer_disease_vs_ADNI_control')].set_index('parcel')
    assert len(n)==len(a)==68 and n.index.is_unique and a.index.is_unique
    assert set(n.index)==set(a.index)
    z=n[['region','hemisphere']].copy()
    z['bvFTD_minus_AD_mm']=n.adjusted_thickness_difference_mm-a.adjusted_thickness_difference_mm
    # Anatomical descriptive partition, chosen post hoc; not a negative-control set.
    frontal={'superiorfrontal','rostralmiddlefrontal','caudalmiddlefrontal',
             'parsopercularis','parsorbitalis','parstriangularis','lateralorbitofrontal',
             'medialorbitofrontal','frontalpole','precentral','paracentral'}
    z['partition']=np.where(z.region.isin(frontal),'Frontal','Other cortex')
    z.to_csv(out/'all_68_parcel_differences.tsv',sep='\t')
    summary=z.groupby('partition').bvFTD_minus_AD_mm.agg(['count','mean','median','min','max'])
    summary['n_negative']=z.groupby('partition').bvFTD_minus_AD_mm.apply(lambda s:int((s<0).sum()))
    summary.to_csv(out/'parcel_distribution_summary.tsv',sep='\t')
    fig,ax=plt.subplots(1,2,figsize=(9,4),gridspec_kw={'width_ratios':[1.3,1]},layout='constrained')
    rank=z.sort_values('bvFTD_minus_AD_mm').reset_index()
    for group,color in [('Frontal','#3566a8'),('Other cortex','#bd7038')]:
        sub=rank[rank.partition.eq(group)]
        ax[0].scatter(sub.index+1,sub.bvFTD_minus_AD_mm,s=27,color=color,label=group)
        v=z.loc[z.partition.eq(group),'bvFTD_minus_AD_mm']
        ax[1].hist(v,bins=np.linspace(-.30,.15,19),histtype='step',linewidth=2,color=color,label=group)
    ax[0].axhline(0,color='0.4',lw=.8);ax[0].set(xlabel='Parcel rank (all 68 parcels)',ylabel='bvFTD minus AD disease–control effect (mm)')
    ax[1].axvline(0,color='0.4',lw=.8);ax[1].set(xlabel='Effect difference (mm)',ylabel='Number of parcels')
    ax[0].legend(frameon=False);ax[0].set_title('A. All parcels');ax[1].set_title('B. Descriptive distribution')
    for a0 in ax:a0.spines[['top','right']].set_visible(False)
    fig.savefig(out/'Supplementary_Figure_S5_parcel_distribution.png',dpi=300)
    fig.savefig(out/'Supplementary_Figure_S5_parcel_distribution.pdf')
    text=Path(args.sensitivity_review).read_text()
    raw=text.split('===== cross_disease_sensitivity =====')[1].split('=====')[0].strip()
    s=pd.read_csv(io.StringIO(raw),sep='\t')
    variants=['BASELINE','AGE_OVERLAP','NIFD_SHARED_PROTOCOLS','NIFD_LARGEST_SHARED_PROTOCOL']
    s=s[s.variant.isin(variants)]
    assert len(s)==96 and s[['variant','outcome','comparison']].duplicated().sum()==0
    s.to_csv(out/'all_12_targeted_outcomes.tsv',sep='\t',index=False)
    print(summary.to_string())

if __name__=='__main__':main()
