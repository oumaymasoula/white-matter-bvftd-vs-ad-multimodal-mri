"""Render manuscript-grounded workflow with Matplotlib; no image generation."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def main():
    out = Path(__file__).resolve().parents[2] / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(11, 12))
    ax.set(xlim=(0, 11), ylim=(0, 12)); ax.axis('off')
    def box(x, y, w, h, text, color='#EAF1F8', dashed=False, size=10):
        ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0.08',
                     linewidth=1.1, edgecolor='#3B5368', facecolor=color,
                     linestyle='--' if dashed else '-'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.5)
    def arrow(a,b):
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#3B5368','lw':1.3})
    ax.text(5.5,11.7,'Multimodal dementia MRI workflow',ha='center',fontsize=18,weight='bold')
    box(.6,10.45,9.8,.8,'NIFD: bvFTD and controls     |     ADNI: AD, MCI and controls\nSelected imaging records; diagnosis and acquisition metadata')
    for x,title in [(.6,'T1-weighted MRI\nFreeSurfer anatomy\nVersions and QC to verify'),
                    (4,'FLAIR MRI\nWMH segmentation\nMethod and masks to verify'),
                    (7.4,'Diffusion MRI\nCorrection and tensor fitting\nCommands and gradients to verify')]:
        box(x,8.75,3,.95,title,dashed=True,color='#FFF3DE',size=9.5)
        arrow((x+1.5,10.45),(x+1.5,9.78))
    for x,title in [(.6,'68 cortical parcels; regional volumes\nThickness and volume composites'),
                    (4,'WMH volume (mL)\nlog(1 + volume) for modelling'),
                    (7.4,'20 tract means: FA, MD, RD, AxD\nBilateral ALPS index')]:
        box(x,7.45,3,.85,title,size=9)
        arrow((x+1.5,8.67),(x+1.5,8.38))
        arrow((x+1.5,7.37),(x+1.5,6.95))
    box(.6,6,9.8,.85,'Common imaging sample\n347 reference participants → 337 complete measurements → 307 site-supported participants')
    box(.6,4.55,6.1,.95,'Primary imaging models\nNIFD: bvFTD + controls; ADNI: AD + MCI + controls\nAdjusted disease–control effects; cross-cohort contrasts; FDR families',size=9.5)
    box(7.25,4.55,3.15,.95,'Clinical matching\nNearest valid assessment ±90 days\nMMSE, CDR; NIFD behavior measures',size=9)
    arrow((3.65,5.92),(3.65,5.58));arrow((8.85,5.92),(8.85,5.58))
    box(.6,3,4.6,.9,'Sensitivity analyses\nAge overlap; NIFD protocol restrictions; site omission\nClinical-severity restrictions',size=9.5)
    box(5.8,3,4.6,.9,'Exploratory analyses\n14 imaging–clinical associations; cortical and tract profiles\nControl-SD scaling and bootstrap intervals',size=9)
    arrow((3,4.47),(3,3.98));arrow((8.3,4.47),(8.3,3.98))
    box(.6,1.55,9.8,.85,'Multimodal synthesis\nShared and distinguishing group-level profiles; native-unit estimates and uncertainty\nNo validated individual diagnostic model')
    arrow((3,2.92),(3,2.48));arrow((8.3,2.92),(8.3,2.48))
    ax.text(.6,.8,'Dashed boxes: implementation details still require source verification.\nEmpty WMH masks remain unresolved; arrows show workflow, not causal relationships.\nAbbreviations are defined in the manuscript. Code-rendered draft for Supplementary Methods.',
            fontsize=9,color='#46505A',va='center',linespacing=1.5)
    fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.98)
    for ext in ['pdf','svg','png']:
        fig.savefig(out/f'pipeline.{ext}',dpi=240,facecolor='white')
    plt.close(fig)

if __name__ == '__main__': main()
