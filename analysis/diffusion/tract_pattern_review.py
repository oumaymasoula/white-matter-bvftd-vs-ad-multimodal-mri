"""Describe all 20 full-sample FA contrasts, without selection filtering.
Usage: python tract_pattern_review.py --effects All_adjusted_effects.tsv --out DIR
Approximate Welch t intervals use independent HC3 variances and the original
full-sample FA model residual degrees of freedom (NIFD 100; ADNI 181).
These are approximate small-sample sensitivities, not exact robust t inference.
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--effects',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    s=pd.read_csv(a.effects,sep=None,engine='python')
    s=s[s.family.eq('DTI')&s.metric.eq('FA')]
    n=s[s.dataset.eq('NIFD')&s.comparison.eq('bvFTD_vs_NIFD_control')].set_index('feature')
    d=s[s.dataset.eq('ADNI')&s.comparison.eq('Alzheimer_disease_vs_ADNI_control')].set_index('feature')
    assert len(n)==len(d)==20 and n.index.is_unique and d.index.is_unique
    assert set(n.index)==set(d.index) and n.model_n.eq(109).all() and d.model_n.eq(198).all()
    z=n[['adjusted_effect','SE_HC3']].join(d[['adjusted_effect','SE_HC3']],lsuffix='_bvFTD',rsuffix='_AD')
    z['difference']=z.adjusted_effect_bvFTD-z.adjusted_effect_AD
    v1=z.SE_HC3_bvFTD**2;v2=z.SE_HC3_AD**2
    z['SE_independent']=np.sqrt(v1+v2)
    z['df_Welch_approx']=(v1+v2)**2/(v1**2/100+v2**2/181)
    crit=stats.t.ppf(.975,z.df_Welch_approx)
    z['CI_low_Welch_approx']=z.difference-crit*z.SE_independent
    z['CI_high_Welch_approx']=z.difference+crit*z.SE_independent
    assert abs(z.loc['Forceps minor | FA','difference']+.0974439551433619)<1e-10
    z.to_csv(out/'all_20_FA_tract_differences.tsv',sep='\t')
    z=z.sort_values('difference');names=z.index.str.replace(' | FA','',regex=False)
    fig,ax=plt.subplots(figsize=(8,7),layout='constrained')
    for i,(name,row) in enumerate(z.iterrows()):
        color='#3566a8' if ('Forceps minor' in name or 'Uncinate' in name) else '#bd7038' if any(w in name for w in ['Corticospinal','Forceps major','Cingulum (hippocampus)']) else '#777777'
        ax.errorbar(row.difference,i,xerr=[[row.difference-row.CI_low_Welch_approx],[row.CI_high_Welch_approx-row.difference]],fmt='o',color=color,capsize=2,markersize=4)
    ax.set_yticks(range(20),names);ax.invert_yaxis();ax.axvline(0,color='0.3',lw=.8)
    ax.set_xlabel('bvFTD minus AD disease–control FA effect (95% approximate Welch CI)')
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'Supplementary_Figure_S6_all_20_FA_tracts.png',dpi=300)
    fig.savefig(out/'Supplementary_Figure_S6_all_20_FA_tracts.pdf')
    print(z[['difference','CI_low_Welch_approx','CI_high_Welch_approx']].to_string())

if __name__=='__main__':main()
