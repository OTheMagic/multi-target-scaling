"""Verify anomalous cached target against the original ARFF source."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'data/real_exps/data/rf2.arff'
found=[]; attrs=[]; index=-1;data=False
with path.open() as f:
    for lineno,line in enumerate(f,1):
        if not data:
            if line.lower().startswith('@attribute'):attrs.append(line.split()[1])
            if line.strip().lower()=='@data':data=True
            continue
        if not line.strip() or line.lstrip().startswith('%'):continue
        index+=1
        if index in range(4779,4786):
            row=next(csv.reader([line]))
            found.append(dict(original_index=index,file_line=lineno,target=row[577],current_feature=row[1],columns=len(row)))
assert attrs[577]=='NASI2_48H__0'
assert float(next(r for r in found if r['original_index']==4782)['target'])==78.9
result=dict(source=str(path),target_column=attrs[577],neighbor_rows=found,status='Cached anomalous value verified in original ARFF; not introduced by the diagnostic cache.')
(ROOT/'data/envelope_method/results/rf2_remove_one/source_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
