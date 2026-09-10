"""One-time retirement of unfitted empirical toy drivers; keep formula helpers."""
import ast
import hashlib
import json
import os
from migrate_fresh_protocol import ROOT, REVIEW, replace_function


def main():
    frozen=ROOT/'envelope_method/toy_designs.json'
    path=ROOT/'tmp/envelope_cqhr_toy.py'
    source=path.read_text(encoding='utf-8')
    if frozen.exists():
        print('Toy entry points already migrated.')
        return
    main_node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    designs=ast.literal_eval(next(n.value for n in main_node.body if isinstance(n,ast.Assign)))
    frozen.write_text(json.dumps(designs,indent=2),encoding='utf-8')
    manifest=[]
    def archive(path,reason):
        if not path.exists():
            return
        src=path.resolve(strict=True)
        dst=(REVIEW/src.relative_to(ROOT)).resolve()
        assert src.is_relative_to(ROOT) and not src.is_relative_to(REVIEW) and dst.is_relative_to(REVIEW)
        if dst.exists():
            raise FileExistsError(dst)
        dst.parent.mkdir(parents=True,exist_ok=True)
        manifest.append(dict(source=str(src.relative_to(ROOT)),destination=str(dst.relative_to(REVIEW)),
                             reason=reason,bytes=src.stat().st_size,sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
        os.rename(src,dst)
    for filename,removed in [('envelope_cqhr_toy.py','run_scenario'),('envelope_cqhr_misspecified_width.py','run')]:
        path=ROOT/'tmp'/filename
        source=path.read_text(encoding='utf-8')
        source=replace_function(source,removed,'# Empirical evaluation moved to the fresh fitted toy runner.')
        source=replace_function(source,'main','''def main():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'envelope_method'))
    from run_toys import cqr_toys
    cqr_toys()
''')
        ast.parse(source)
        archive(path,'Superseded unfitted empirical toy driver; mathematical helper functions retained in active code.')
        path.write_text(source,encoding='utf-8')
    for name in ['envelope_cqhr_toy_summary.json','envelope_cqhr_misspecified_width_summary.json',
                 'signed_lwc_toy_region.png','signed_lwc_toy_region.svg','signed_lwc_toy_summary.json']:
        archive(ROOT/'tmp'/name,'Superseded score-only empirical result, without fresh fitted training model.')
    archive(ROOT/'envelope_method/inventory.json','Pre-migration inventory contains superseded source code and data provenance.')
    (REVIEW/'additional_manifest.json').write_text(json.dumps(dict(status='complete',files=manifest),indent=2),encoding='utf-8')
    print('Retired unfitted toy entry points; archived',len(manifest),'additional files.')


if __name__=='__main__':
    main()
