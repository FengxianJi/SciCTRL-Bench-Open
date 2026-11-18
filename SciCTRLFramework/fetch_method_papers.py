"""Download the five newly selected methods from arXiv and extract local MD."""
import json
from pathlib import Path
import urllib.request
import fitz

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'_ICLR27_SciCTRL_Bench__A_Benchmark__For_End_to_End_AI_Scientists_Controlled_Feedback_Revision_Evaluation'
SELECT={'autoresearchclaw2026','ma2026sage','yamada2025scientistv2','evoscientist2026','internagent2026'}


if __name__=='__main__':
    metadata=json.loads((PAPER/'bibliography_sources.json').read_text())
    for entry in metadata:
        if entry['key'] not in SELECT: continue
        identifier=entry['url'].rsplit('/',1)[-1]
        pdf=ROOT/'paper'/('arxiv_'+identifier+'.pdf')
        if not pdf.exists():
            request=urllib.request.Request('https://arxiv.org/pdf/'+identifier,headers={'User-Agent':'SciCTRL local literature verification'})
            with urllib.request.urlopen(request,timeout=90) as response:
                data=response.read()
            if not data.startswith(b'%PDF'):raise ValueError('not a PDF')
            pdf.write_bytes(data)
        doc=fitz.open(pdf)
        title=entry['metadata']['citation_title'][0]
        parts=['# '+title,'Source: '+entry['url'],
               'Text extraction from the original PDF; tables/equations may need visual checking.']
        parts.extend('<!-- Page '+str(i+1)+' -->\n'+page.get_text() for i,page in enumerate(doc))
        pdf.with_suffix('.md').write_text('\n\n'.join(parts))
        print(entry['key'],len(doc),'pages',flush=True)
