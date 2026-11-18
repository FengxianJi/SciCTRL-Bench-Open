"""Inspect compiled manuscript text and render representative review pages."""
import argparse
import json
from pathlib import Path
import pymupdf as fitz

p=argparse.ArgumentParser()
p.add_argument('pdf',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
doc=fitz.open(a.pdf)
texts=[page.get_text() for page in doc]
selected={0}
for phrase in ['Blind-review measurements','Every field in the executable public schema','Paired diagnosis-given ablation']:
    found=next((i for i,t in enumerate(texts) if phrase.lower() in ' '.join(t.split()).lower()),None)
    if found is not None:selected.add(found)
renders=[]
for index in sorted(selected):
    target=a.output/f'page-{index+1}.png'
    doc[index].get_pixmap(matrix=fitz.Matrix(1.4,1.4)).save(target)
    renders.append(str(target.resolve()))
report={'pages':len(doc),'unresolved_reference_pages':[i+1 for i,t in enumerate(texts) if '??' in t],
        'review_images':renders,'pdf':str(a.pdf.resolve())}
(a.output/'inspection.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
