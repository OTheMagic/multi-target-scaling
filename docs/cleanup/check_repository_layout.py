"""Check the source/data boundary and prospective GitHub file sizes, read-only."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from docs.cleanup.centralize_data import candidates


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8').split('\0')


def check():
    outside = list(candidates())
    visible = set(git('ls-files', '--cached', '--others', '--exclude-standard', '-z')) - {''}
    existing = [ROOT / name for name in visible if (ROOT / name).is_file()]
    large = [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size)
             for p in existing if p.stat().st_size >= 100_000_000]
    still_tracked = [name for name in git('ls-files', '-ci', '--exclude-standard', '-z') if name]
    ignored = {}
    for folder in ('data', 'tmp'):
        ignored[folder] = subprocess.run(
            ['git', 'check-ignore', '--no-index', '-q', f'{folder}/layout-probe'],
            cwd=ROOT).returncode == 0
    return dict(status='passed' if not outside and not large and not still_tracked
                and all(ignored.values()) else 'failed',
                research_data_outside_data=outside, oversized_git_files=large,
                ignored_files_still_tracked=still_tracked, ignored_roots=ignored,
                prospective_working_tree_files=len(existing),
                prospective_working_tree_bytes=sum(p.stat().st_size for p in existing),
                data_available=(ROOT / 'data/envelope_method/results').exists(),
                note='Checks working files and index; does not rewrite Git history, commit or push.')


if __name__ == '__main__':
    result = check()
    print(json.dumps(result, indent=2))
    raise SystemExit(result['status'] != 'passed')
