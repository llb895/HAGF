"""Exploratory enrichment of archived auxiliary model masks in KEGG cancer pathways."""
import os
from pathlib import Path
import json, hashlib, re
import requests
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(
    os.environ.get("HAGF_PROJECT_ROOT", Path(__file__).resolve().parents[1])
)
SRC=ROOT/'Methylation/SUBJECT/HRA003209/KEGG'
OUT=ROOT/'revision_workspace/response_evidence/organ_knowledge_consistency'
OUT.mkdir(parents=True,exist_ok=True)
CANCERS=['BRCA','COREAD','ESCA','STAD','LIHC','NSCLC','PACA']
PATHWAYS={'BRCA':'hsa05224','COREAD':'hsa05210','STAD':'hsa05226','LIHC':'hsa05225','NSCLC':'hsa05223','PACA':'hsa05212'}
sets={};provenance=[]
for cancer,pid in PATHWAYS.items():
    dest=OUT/(pid+'.txt')
    if not dest.exists():
        r=requests.get('https://rest.kegg.jp/get/'+pid,timeout=30);r.raise_for_status();dest.write_text(r.text)
    text=dest.read_text(); field=''; genes=set()
    for line in text.splitlines():
        if line[:12].strip():field=line[:12].strip()
        if field=='GENE':
            m=re.match(r'\s*\d+\s+([^;]+);',line[12:])
            if m: genes.add(m.group(1).strip().split(',')[0].upper())
    assert len(genes)>20,(pid,len(genes))
    sets[cancer]=genes
    provenance.append({'pathway':pid,'gene_count':len(genes),'url':'https://rest.kegg.jp/get/'+pid,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})

# The annotation universe is restricted explicitly to genes archived in this project.
annotation=pd.concat([pd.read_csv(p) for p in sorted((SRC/'KEGG_info').glob('*_KEGG_GO_info.csv'))]).drop_duplicates(['gene_id','chr','start','end'])
annotation['gene_name']=annotation.gene_name.astype(str).str.upper()
annotation=annotation[~annotation.gene_name.str.startswith('ENSG')]
genes_by_chr={c:g for c,g in annotation.groupby('chr')}
records=[]; sources=[]
for modality,prefix in [('CNV','copy_number'),('Methylation','methy')]:
    mapping=None;expected=None
    for cancer in CANCERS:
        paths=[SRC/f'{prefix}_{cancer}_mask_{i}.csv' for i in (1,2)]
        frames=[pd.read_csv(p) for p in paths]
        assert list(frames[0].columns)==list(frames[1].columns)
        # Retain every correctly named autosomal region; do not discard the first feature.
        columns=[c for c in frames[0].columns if re.fullmatch(r'(?:chr)?(?:[1-9]|1\d|2[0-2])-\d+(?:\.\d+)?-\d+(?:\.\d+)?',c)]
        assert len(columns)>20
        if expected is not None:assert columns==expected
        expected=columns
        arr=pd.concat(frames,ignore_index=True)[columns].to_numpy(float)
        assert np.isfinite(arr).all() and (arr>=0).all()
        # Normalize each mask before averaging so each stored mask contributes equally.
        totals=arr.sum(axis=1); assert (totals>0).all()
        score=(arr/totals[:,None]).mean(axis=0)
        if mapping is None:
            mapping=[]
            for col in columns:
                chrom,start,end=col.replace('chr','').split('-'); start,end=int(float(start)),int(float(end))
                g=genes_by_chr.get('chr'+chrom)
                hits=set() if g is None else set(g.loc[(g.start<=end)&(g.end>=start),'gene_name'])
                mapping.append(hits)
        n=int(np.ceil(.2*len(columns)))
        idx=np.argsort(-score,kind='stable')[:n]
        selected=set().union(*(mapping[i] for i in idx))
        universe=set().union(*mapping)
        pd.DataFrame({'region':columns,'weight':score,'selected':[i in set(idx) for i in range(len(columns))]}).to_csv(OUT/f'{prefix}_{cancer}_regions.csv',index=False)
        sources.append({'modality':modality,'cancer':cancer,'files':[str(p) for p in paths],
                        'sha256':[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
                        'mask_rows':[len(f) for f in frames],'regions':len(columns),'selected_regions':n,
                        'mapped_background_genes':len(universe),'selected_genes':len(selected)})
        for target,pid in PATHWAYS.items():
            known=sets[target]&universe
            a=len(selected&known);b=len(selected-known);c=len(known-selected);d=len(universe-(selected|known))
            _,p=fisher_exact([[a,b],[c,d]],alternative='greater')
            odds=((a+.5)*(d+.5))/((b+.5)*(c+.5))
            records.append({'modality':modality,'model_cancer':cancer,'reference_cancer':target,'pathway':pid,
                            'overlap':a,'selected_genes':len(selected),'reference_genes':len(known),
                            'background_genes':len(universe),'odds_ratio_corrected':odds,'p':p,
                            'overlap_genes':';'.join(sorted(selected&known))})
        print(modality,cancer,'complete',flush=True)
df=pd.DataFrame(records);df['q']=multipletests(df.p,method='fdr_bh')[1];df.to_csv(OUT/'enrichment.csv',index=False)
(OUT/'provenance.json').write_text(json.dumps({'references':provenance,'sources':sources,
 'interpretation':'Exploratory archived auxiliary single-profile cancer-versus-healthy model masks; not frozen final HAGF class attributions.',
 'background':'Genes in retained archived annotations overlapping all eligible model regions; incomplete genomic annotation coverage is possible.',
 'test':'One-sided Fisher exact; BH across all 84 tests; no claim that shared cancer pathways establish organ specificity.'},indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(11,6.3));fig.subplots_adjust(left=.08,right=.90,bottom=.24,top=.88,wspace=.26)
targets=list(PATHWAYS)
limit=max(1,float(np.ceil(np.abs(np.log2(df.odds_ratio_corrected)).max())) )
for ax,modality in zip(axes,['CNV','Methylation']):
    sub=df[df.modality==modality]
    mat=sub.pivot(index='model_cancer',columns='reference_cancer',values='odds_ratio_corrected').loc[CANCERS,targets]
    qs=sub.pivot(index='model_cancer',columns='reference_cancer',values='q').loc[CANCERS,targets]
    im=ax.imshow(np.log2(mat),vmin=-limit,vmax=limit,cmap='RdBu_r',aspect='auto')
    ax.set_title('Bie et al. dataset - '+modality,fontsize=11,fontweight='bold',pad=12)
    ax.set_xticks(range(6),targets,rotation=40,ha='right');ax.set_yticks(range(7),CANCERS)
    for i,cancer in enumerate(CANCERS):
        for j,target in enumerate(targets):
            value=np.log2(mat.iloc[i,j]);star='*' if qs.iloc[i,j]<.05 else ''
            ax.text(j,i,f'{value:.2f}{star}',ha='center',va='center',fontsize=8,color='white' if abs(value)>.6*limit else 'black')
            if cancer==target:ax.add_patch(Rectangle((j-.49,i-.49),.98,.98,fill=False,ec='black',lw=1.6))
axes[0].set_ylabel('Cancer-specific auxiliary model')
fig.colorbar(im,cax=fig.add_axes([.925,.24,.016,.64]),label='log2 odds ratio')
fig.text(.08,.135,'Columns: KEGG cancer pathways. Black outlines: matching cancer. No test reached BH-adjusted P < 0.05.',fontsize=8.3)
fig.text(.08,.09,'Background: archived annotated genes covered by each profile. Shared pathway genes limit claims of organ specificity.',fontsize=8.3)
fig.text(.08,.045,'Exploratory historical auxiliary models; ESCA has no corresponding pathway in this six-pathway reference panel.',fontsize=8.3)
fig.savefig(OUT/'HAGF_organ_knowledge_consistency.pdf');plt.close(fig)
print(df[df.model_cancer==df.reference_cancer][['modality','model_cancer','odds_ratio_corrected','q']].to_string(index=False),flush=True)
print('ORGAN_KNOWLEDGE_COMPLETE',flush=True)
