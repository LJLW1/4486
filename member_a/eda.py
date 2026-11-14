"""Reproducible EDA for the repository's main training data, no model preprocessing."""
import os,json,hashlib
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cs4486-member-a-repo-mpl")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import TARGETS
TARGETS = list(TARGETS)
from member_a.run import write_json as dump
from common.prepare import sha256 as sha
COLORS = ["#375B82", "#2A9D8F", "#E9B44C", "#C96B5C"]

def audit(raws, data, out):
    stats, target_rows = {}, []
    for name, frame in raws.items():
        stats[name] = {"rows": len(frame), "columns": len(frame.columns),
                       "patients": int(frame.patient_id.nunique()), "visits": int(frame.visit_id.nunique()),
                       "months": sorted(frame.visit_month.unique().tolist()),
                       "missing_counts": {c: int(v) for c,v in frame.isna().sum().items()},
                       "sha256": sha(data/name)}
        if "UniProt" in frame:
            stats[name]["unique_proteins"] = int(frame.UniProt.nunique())
        if "Peptide" in frame:
            stats[name]["unique_peptides"] = int(frame.Peptide.nunique())
    c=raws["train_clinical_data.csv"]
    p=raws["train_proteins.csv"]
    q=raws["train_peptides.csv"]
    for t in TARGETS:
        y=c[t].dropna()
        target_rows.append(dict(target=t, observed=len(y), missing=int(c[t].isna().sum()),
                                missing_percent=100*c[t].isna().mean(), zero_percent_observed=100*y.eq(0).mean(),
                                mean=y.mean(), median=y.median(), std=y.std(), minimum=y.min(), maximum=y.max()))
    checks={"duplicate_clinical_patient_month":int(c.duplicated(["patient_id","visit_month"]).sum()),
            "duplicate_protein_visit_id_UniProt":int(p.duplicated(["visit_id","UniProt"]).sum()),
            "duplicate_peptide_visit_id_Peptide":int(q.duplicated(["visit_id","Peptide"]).sum()),
            "same_protein_peptide_visit_sets":set(p.visit_id)==set(q.visit_id),
            "clinical_visits_with_proteins":int(c.visit_id.isin(p.visit_id).sum()),
            "main_supplemental_patient_overlap":len(set(c.patient_id)&set(raws['supplemental_clinical_data.csv'].patient_id))}
    for k in ['duplicate_clinical_patient_month','duplicate_protein_visit_id_UniProt','duplicate_peptide_visit_id_Peptide']:
        if checks[k]: raise ValueError(f"Audit needs manual review: {k}")
    dump(out/"audit.json", {"tables":stats,"checks":checks})
    pd.DataFrame(target_rows).to_csv(out/"target_statistics.csv",index=False)
    prof=pd.DataFrame([{**{"file":name},**{k:v for k,v in info.items() if k in ['rows','columns','patients','visits','unique_proteins','unique_peptides']}} for name,info in stats.items()])
    prof.to_csv(out/"data_profile.csv",index=False)
    # A null-free long table can still produce missing cells after reshaping.
    for kind,frame,key in [('protein',p,'UniProt'),('peptide',q,'Peptide')]:
        counts=frame.groupby('visit_id')[key].nunique()
        stats[kind+"_panel"]={"possible_cells":int(len(counts)*frame[key].nunique()),"observed_cells":len(frame),
                              "wide_missing_percent":float(100*(1-len(frame)/(len(counts)*frame[key].nunique()))),
                              "min_measured_per_visit":int(counts.min()),"median_measured_per_visit":float(counts.median())}
    dump(out/"audit.json", {"tables":stats,"checks":checks})
    return stats

def eda(raws, out):
    figures=out/"figures";figures.mkdir()
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,
                         "axes.spines.right":False,"figure.facecolor":"white","axes.titleweight":"bold"})
    c,p,q=[raws[n] for n in ["train_clinical_data.csv","train_proteins.csv","train_peptides.csv"]]
    fig,axes=plt.subplots(1,2,figsize=(12,4.1),layout="constrained")
    for frame,label,color in [(c,'Clinical',COLORS[0]),(p.drop_duplicates('visit_id'),'Protein / peptide',COLORS[1])]:
        axes[0].hist(frame.groupby('patient_id').size(),bins=np.arange(.5,24.5,1),alpha=.65,label=label,color=color)
    axes[0].set(xlabel="Visits per patient",ylabel="Patients",title="Repeated and uneven follow-up");axes[0].legend(frameon=False)
    counts=pd.DataFrame({'Clinical':c.groupby('visit_month').size(),'Protein / peptide':p.drop_duplicates('visit_id').groupby('visit_month').size()}).fillna(0)
    counts.plot.bar(ax=axes[1],color=COLORS[:2],width=.8)
    axes[1].set(xlabel="Month from baseline",ylabel="Visits",title="Clinical and blood sampling differ");axes[1].legend(frameon=False)
    fig.savefig(figures/'01_visits.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.1),layout="constrained")
    missing=100*c[TARGETS].isna().mean();zero=[100*c[t].dropna().eq(0).mean() for t in TARGETS]
    x=np.arange(4)
    axes[0].bar(x-.2,missing,.38,label='Missing / all rows',color=COLORS[0])
    axes[0].bar(x+.2,zero,.38,label='Zero / observed labels',color=COLORS[1])
    axes[0].set(xticks=x,xticklabels=TARGETS,ylabel='Percent',title='Missing labels and true zeros differ')
    axes[0].legend(frameon=False,fontsize=9)
    for i,t in enumerate(TARGETS):
        axes[1].step(np.sort(c[t].dropna()),np.arange(1,c[t].notna().sum()+1)/c[t].notna().sum(),label=t,color=COLORS[i])
    axes[1].set(xlabel='Observed score',ylabel='Cumulative fraction',title='Targets have different scales');axes[1].legend(frameon=False)
    fig.savefig(figures/'02_targets.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.1),layout='constrained')
    for i,(frame,key) in enumerate([(p,'NPX'),(q,'PeptideAbundance')]):
        groups=frame.groupby('visit_id').size()
        axes[i].hist(groups,bins=22,color=COLORS[i],edgecolor='white')
        axes[i].set(xlabel='Measured proteins' if i==0 else 'Measured peptides',ylabel='Blood visits',
                    title=f"{'Protein' if i==0 else 'Peptide'} panel coverage varies")
    fig.savefig(figures/'03_panel_coverage.png',dpi=200);plt.close(fig)

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--data-dir", default="amp-parkinsons-disease-progression-prediction");p.add_argument("--output-dir",default="member_a/eda")
    a=p.parse_args();data=Path(a.data_dir);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False)
    names=["train_clinical_data.csv","train_proteins.csv","train_peptides.csv","supplemental_clinical_data.csv"]
    raws={n:pd.read_csv(data/n) for n in names}
    audit(raws,data,out);eda(raws,out)

if __name__=="__main__":main()
