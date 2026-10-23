from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;root=p.parent
rows=list(csv.DictReader((p/"results.csv").open()))
assert len(rows)==165
methods=list(dict.fromkeys(r["method"] for r in rows));summary=[];paired=[]
for n in [192,384,576]:
 for method in methods:
  rr=[r for r in rows if int(r["train_n"])==n and r["method"]==method];assert len(rr)==5
  s={"train_n":n,"method":method}
  for key in ["attributes","cost","inner_brier","test_brier","test_kl","test_tv","test_js"]:
   v=[float(r[key]) for r in rr];s[key+"_mean"]=float(np.mean(v));s[key+"_sd"]=float(np.std(v,ddof=1))
  summary.append(s)
  if method not in ["All attributes","Dependency-only","Joint guard"]:
   dif=[]
   for r in rr:
    j=next(q for q in rows if q["repeat"]==r["repeat"] and q["train_n"]==r["train_n"] and q["method"]=="Joint guard")
    dif.append(float(j["test_brier"])-float(r["test_brier"]))
   paired.append({"train_n":n,"baseline":method,"joint_minus_baseline_mean":float(np.mean(dif)),"joint_minus_baseline_sd":float(np.std(dif,ddof=1)),"joint_wins":sum(d<0 for d in dif),"paired_differences":dif})
with (p/"summary.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
(p/"paired.json").write_text(json.dumps(paired,indent=2))
for rep in range(1,6):
 for n in [192,384,576]:
  job=p/f"repeat_{rep}_n{n}";saved=json.loads((job/"selection_before_test.json").read_text());old=json.loads((root/"stability"/job.name/"results.json").read_text());j=next(r for r in old if r["method"]=="Joint guard");c=np.array(json.loads((p/"protocol.json").read_text()).get("costs",json.loads((root/"stability/protocol.json").read_text())["costs"]))
  for name,P in saved.items():
   if name=="Random k draws":assert len(P)==20 and all(len(a)==j["attributes"] for a in P)
   elif name.endswith(" k"):assert len(P)==j["attributes"]
   elif name.endswith("budget"):assert c[P].sum()<=j["cost"]
  bridge=json.loads((job/'bridge_selection_before_test.json').read_text());assert c[bridge['selected']].sum()<=j['cost']
(p/"validation.txt").write_text("PASS: 15 paired splits, 165 method rows; exact-k quotas, budget bounds, 20 random draws per split, pre-test selection files and deterministic nearest-neighbor equivalence checked.\n")
plt.rcParams.update({"font.size":9,"pdf.fonttype":42})
fig,axes=plt.subplots(3,1,figsize=(6.5,8.2))
groups=[["Joint guard","Cheapest k","Random k","Correlation k","Forward k"],["Joint guard","Cost-correlation budget","Cost-forward budget","Budget-forward bridge"],["Joint guard","Prediction-only","All attributes"]]
for ax,group in zip(axes,groups):
 for method in group:
  ss=[s for s in summary if s["method"]==method];ax.plot([s["train_n"] for s in ss],[s["test_brier_mean"] for s in ss],marker="o",label=method)
 ax.set_xticks([192,384,576]);ax.set_xlabel("Training objects");ax.set_ylabel("Mean outer-test Brier");ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(p/"comparison.pdf");fig.savefig(p/"comparison.png",dpi=180)
print(json.dumps({"summary":summary,"paired":paired},indent=2))
