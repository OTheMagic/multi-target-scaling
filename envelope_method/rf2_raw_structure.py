"""Read original ARFF to audit missingness and temporal label consistency."""
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
rows=[];data=False
with (ROOT/'data/real_exps/data/rf2.arff').open() as f:
    for line in f:
        if line.strip().lower()=='@data':data=True;continue
        if not data or not line.strip() or line.lstrip().startswith('%'):continue
        rows.append([np.nan if v.strip()=='?' else float(v) for v in next(csv.reader([line]))])
a=np.asarray(rows)
ok=np.isfinite(a).all(1);missing_target=~np.isfinite(a[:,576:]).all(1)
details=[]
for j in range(8):
    t=a[:-48,576+j];future=a[48:,j];valid=np.isfinite(t)&np.isfinite(future)
    bad=np.flatnonzero(valid&~np.isclose(t,future,rtol=0,atol=1e-8))
    details.append(dict(coordinate=j,comparable_pairs=int(valid.sum()),mismatches=len(bad),first_mismatched_indices=bad[:10].tolist()))
out=dict(raw_rows=len(a),complete_rows=int(ok.sum()),dropped_rows=int((~ok).sum()),rows_with_missing_targets=int(missing_target.sum()),rows_with_only_missing_features=int((~ok&~missing_target).sum()),anomalous_target=float(a[4782,577]),corresponding_future_input=None if not np.isfinite(a[4830,1]) else float(a[4830,1]),future_row_has_missing_fields=bool(not ok[4830]),future_row_missing_columns=np.flatnonzero(~np.isfinite(a[4830])).tolist(),lag_consistency=details)
(ROOT/'data/envelope_method/results/rf2_remaining/raw_structure.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
