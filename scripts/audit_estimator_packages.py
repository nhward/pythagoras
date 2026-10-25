"""Refresh primary discovery sources and report package/estimator audit coverage.

Network access is opt-in. Refreshing discovers candidates; it never installs
packages or automatically approves them for estimator inspection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parents[1] / 'app/data/estimator_capabilities'
SOURCES = {
    'related_projects': 'https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/related_projects.rst',
    'contrib_readme': 'https://raw.githubusercontent.com/scikit-learn-contrib/scikit-learn-contrib/master/README.md',
    'contrib_repositories': 'https://api.github.com/orgs/scikit-learn-contrib/repos?per_page=100&type=public',
}


def extract_projects(kind, content):
    if kind == 'related_projects':
        # Top-level list entries only: omit inline explanatory links and the
        # explicitly out-of-scope documentation translations section.
        content = content.split('Translations of scikit-learn documentation')[0]
        return [{'name': ' '.join(name.split()), 'url': url}
                for name, url in re.findall(r'^- `([^`<>]+?)\s*<(https?://[^>]+)>`_', content, re.MULTILINE)]
    if kind == 'contrib_readme':
        return [{'name': name, 'url': url}
                for name, url in re.findall(r'^## \[([^]]+)\]\((https?://[^)]+)\)', content, re.MULTILINE)]
    return [{'name': repo['name'], 'url': repo['html_url'], 'archived': repo['archived']}
            for repo in json.loads(content)]


def refresh():
    snapshots = []
    for kind, url in SOURCES.items():
        pages, projects = [], []
        next_url = url
        while next_url:
            request = urllib.request.Request(next_url, headers={'User-Agent': 'Pythagoras-package-audit'})
            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read()
                links = response.headers.get('Link', '')
            pages.append({'url': next_url, 'sha256': hashlib.sha256(content).hexdigest()})
            projects.extend(extract_projects(kind, content.decode()))
            match = re.search(r'<([^>]+)>; rel="next"', links)
            next_url = match[1] if match else None
        if not projects:
            raise ValueError(f'No projects extracted from {url}; inspect source format changes')
        snapshots.append({'id': kind, 'url': url, 'retrieved_at': datetime.now(UTC).isoformat(),
                          'pages': pages, 'projects': projects})
    return {'scope': 'All top-level project entries in Related Projects before translations, all contrib README project headings, and all public contrib repositories (including archived/infrastructure repositories). Broader directories and method-family searches remain future audit phases.',
            'sources': snapshots}


def report(registry, snapshot, catalogue):
    counts = Counter(row['status'] for row in registry['packages'])
    covered = {tuple(ref) for row in registry['packages'] for ref in row['source_entries']}
    unreviewed = [(source['id'], project['name']) for source in snapshot['sources']
                  for project in source['projects'] if (source['id'], project['name']) not in covered]
    inventory = Counter(entry['package'] for entry in catalogue['estimators'])
    validation = Counter(profile['validation'] for profile in catalogue['profiles'])
    lines = ['# Estimator package audit', '',
             f"Generated: {datetime.now(UTC).isoformat()}", '', snapshot['scope'], '',
             f"Registry: {len(registry['packages'])} projects; decisions: {dict(sorted(counts.items()))}.",
             f"Unreviewed source entries: {len(unreviewed)}.",
             f"Catalogue: {len(catalogue['estimators'])} estimator aliases, {len(catalogue['profiles'])} reviewed profiles, {len(catalogue['specialists'])} documentation-only recipes.",
             f"Profile validation: {dict(sorted(validation.items()))}.", '',
             'Inspection means public-class/signature/tag discovery, not verified pipeline compatibility. Aliases can identify the same implementation. Missing optional packages remain visible in package metadata; their profiles are unavailable.', '',
             '## Source coverage', '', '| Source | Entries | Retrieved |', '|---|---:|---|']
    for source in snapshot['sources']:
        lines.append(f"| [{source['id']}]({source['url']}) | {len(source['projects'])} | {source['retrieved_at']} |")
    lines += ['', '## Package decisions and inspection', '',
              '| Project | Decision | Distribution | Inspection | Estimators | Reason |', '|---|---|---|---|---:|---|']
    for row in registry['packages']:
        package = row['distribution']
        state = catalogue['packages'].get(package, {}).get('status', 'not_requested')
        reason = row['reason'].replace('|', '/')
        lines.append(f"| [{row['id']}]({row['project_url']}) | {row['status']} | {package or '—'} | {state} | {inventory[package]} | {reason} |")
    superseded = [entry for entry in catalogue['estimators'] if entry.get('superseded_by')]
    lines += ['', '## Method grouping', '',
              f'{len(superseded)} methods have reviewed supersession metadata. Linked recipes are hidden by default and restored by Show excluded and unreviewed recipes. This is display grouping; objective-family nesting does not guarantee interchangeable solvers, capabilities or validated recipes.', '',
              '| Method | Broader method | Relationship |', '|---|---|---|']
    for entry in superseded:
        relation = entry['superseded_by']
        lines.append(f"| {entry['id']} | {relation['method']} | {relation['relationship']} |")
    lines += ['', '## Discovery failures', '']
    for issue in catalogue['discovery_issues']:
        lines.append(f"- `{issue['module']}`: {issue['error']}")
    if not catalogue['discovery_issues']:
        lines.append('None recorded.')
    lines += ['', '## New source entries awaiting a decision', '']
    lines.extend(f'- {source}: {name}' for source, name in unreviewed)
    if not unreviewed:
        lines.append('None in this source snapshot.')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--catalogue', type=Path, default=DIRECTORY / 'catalogue.json')
    args = parser.parse_args()
    path = DIRECTORY / 'audit_sources.json'
    if args.refresh:
        path.write_text(json.dumps(refresh(), indent=2) + '\n')
    snapshot = json.loads(path.read_text())
    registry = json.loads((DIRECTORY / 'packages.json').read_text())
    catalogue = json.loads(args.catalogue.read_text())
    (DIRECTORY / 'AUDIT.md').write_text(report(registry, snapshot, catalogue))


if __name__ == '__main__':
    main()
