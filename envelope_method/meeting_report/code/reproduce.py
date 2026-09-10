"""Reproduce this relocated report using numeric evidence under root data/."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--regenerate',action='store_true',help='Also regenerate every absolute trial from its seed and check scores.')
    args=parser.parse_args()
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    print('Numeric evidence: '+str(ROOT.parents[1]/'data/envelope_method/meeting_report'))
    steps=[['code/absolute_experiments.py']+(['--regenerate'] if args.regenerate else []),
           ['code/signed_experiments.py'],['qa/audit_core.py'],['qa/audit_results.py']]
    for step in steps:
        subprocess.run([sys.executable,*step],cwd=ROOT,check=True)
    print('Data, figures, and independent audits reproduced. Compile report.tex to update the PDF.')
