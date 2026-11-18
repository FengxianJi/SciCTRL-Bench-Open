"""Fetch citation metadata from primary arXiv pages; keep retrieval provenance."""
import argparse
import json
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

REFERENCES = {
    'huang2023mlagentbench': '2310.03302',
    'chan2024mlebench': '2410.07095',
    'chen2024scienceagentbench': '2410.05080',
    'starace2025paperbench': '2504.01848',
    'bragg2025astabench': '2510.21652',
    'garikaparthi2026researchgym': '2602.15112',
    'li2026beyond': '2608.13417',
    'xu2026swescience': '2608.19799',
    'autoresearchclaw2026': '2605.20025',
    'ma2026sage': '2606.31478',
    'yamada2025scientistv2': '2504.08066',
    'evoscientist2026': '2603.08127',
    'tang2025airesearcher': '2505.18705',
    'internagent2026': '2602.08990',
}

class Metadata(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'meta' and a.get('name', '').startswith('citation_'):
            self.values.setdefault(a['name'], []).append(a.get('content', ''))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paper_dir', type=Path)
    args = parser.parse_args()
    records, entries = [], []
    for key, identifier in REFERENCES.items():
        url = f'https://arxiv.org/abs/{identifier}'
        request = urllib.request.Request(url, headers={'User-Agent': 'SciCTRL bibliography verification/1.0'})
        with urllib.request.urlopen(request, timeout=45) as response:
            page = response.read().decode()
        meta = Metadata()
        meta.feed(page)
        m = meta.values
        title = m['citation_title'][0]
        authors = m['citation_author']
        year = m['citation_date'][0][:4]
        def tex(s):
            return s.replace('&', r'\&').replace('_', r'\_').replace('%', r'\%')
        entries.append('@article{' + key + ',\n'
            + '  title = {{' + tex(title) + '}},\n'
            + '  author = {' + tex(' and '.join(authors)) + '},\n'
            + f'  year = {{{year}}},\n  journal = {{arXiv preprint arXiv:{identifier}}},\n'
            + f'  eprint = {{{identifier}}},\n  url = {{{url}}}\n' + '}\n')
        records.append({'key': key, 'url': url, 'metadata': m})
        print(key, identifier, flush=True)
    target = args.paper_dir / 'iclr2027_conference.bib'
    if target.exists() and not (args.paper_dir / 'bibliography.original.bib').exists():
        (args.paper_dir / 'bibliography.original.bib').write_bytes(target.read_bytes())
    target.write_text('\n'.join(entries))
    (args.paper_dir / 'bibliography_sources.json').write_text(json.dumps(records, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
