"""Targeted highlighting — only the specific quotes we adopted from each paper.

For each core methodology paper, ~1-5 SHORT, SPECIFIC text snippets are
highlighted in Boilermaker Gold (#CEB888). These are the exact passages we
reference in our paper draft as the source of each methodological choice.

Skips peripheral / irrelevant papers entirely.

Usage:
    python scripts/05_visualization/highlight_papers_targeted.py pdfs/
"""
import sys
from pathlib import Path
import fitz   # PyMuPDF


GOLD_RGB = (0.808, 0.722, 0.533)   # #CEB888

# Each entry = (paper filename pattern, list of TARGETED phrases to highlight)
# Phrases must be short (5-25 words) and unique enough to find via fuzzy search.
TARGETED = {
    'bandaru_2020_phenocrop': {
        'description': 'Bandaru 2020 — PhenoCrop foundation we extend',
        'phrases': [
            # Core APTT definition + 3-component framework
            'accumulated photo-thermal time',
            'PhenoCrop framework constitutes three components',
            'plant phenology model',
            'temperature and photoperiod response functions',
            'Kalman Filter Recursive Downscaling',
        ],
    },
    'wang_engel_1998': {
        'description': 'Wang & Engel 1998 — beta function f(T) we use',
        'phrases': [
            # The temperature beta-function formulation
            'beta function',
            'cardinal temperatures',
            'three sub-phases',
            'vernalization function',
            'photoperiod function',
            'developmental rate',
        ],
    },
    'streck_2003_vernalization': {
        'description': 'Streck 2003 — vernalization function f(V) we adopt',
        'phrases': [
            # The exact vernalization equation we use in WES
            'generalized vernalization response',
            'modified Wang and Engel',
            'vernalization days',
            'cardinal temperatures for vernalization',
        ],
    },
    'porter_gawith_1999': {
        'description': 'Porter & Gawith 1999 — wheat cardinal temperatures',
        'phrases': [
            # The cardinal-temperature numbers we use (1.3, 4.9, 15.7°C)
            'vernalization',
            'cardinal temperatures',
            'base temperature',
            'optimum temperature',
        ],
    },
    'mcmaster_wilhelm_1997': {
        'description': 'McMaster & Wilhelm 1997 — GDD method 2 we use',
        'phrases': [
            # The Method 2 GDD calculation we adopted
            'Method 2',
            'one equation, two interpretations',
            'growing degree-days',
            'upper threshold',
        ],
    },
    'lobert_2023': {
        'description': 'Lobert 2023 — DL benchmark we compare against',
        'phrases': [
            # The specific R² claims we cite + architecture
            'temporal U-Net',
            'phenological stages of winter wheat',
            'Sentinel-1, Sentinel-2',
            'absolute error of less than six days',
        ],
    },
    'zhao_2025_australian_wheat': {
        'description': 'Zhao 2025 — Australian SOTA we compare against',
        'phrases': [
            # The specific R² + method claims we cite
            'Sentinel-2',
            'GLAI',
            'CCC',
            'Australian grain',
            'flag leaf',
            'flowering',
        ],
    },
    # Skip: irigireddy_satflow, nandan_crop_rotations, streck_chronology,
    # shahhosseini_apsim_ml — not directly cited as adopted methodology.
}


def find_paper_key(filename: str) -> str | None:
    stem = Path(filename).stem.lower()
    for k in TARGETED:
        if k in stem:
            return k
    # Fuzzy fallback
    if 'bandaru' in stem and 'phenocrop' in stem:        return 'bandaru_2020_phenocrop'
    if 'wang' in stem and 'engel' in stem:                return 'wang_engel_1998'
    if 'streck' in stem and 'vernalization' in stem:     return 'streck_2003_vernalization'
    if 'porter' in stem or 'gawith' in stem:             return 'porter_gawith_1999'
    if 'mcmaster' in stem or 'wilhelm' in stem:          return 'mcmaster_wilhelm_1997'
    if 'lobert' in stem:                                  return 'lobert_2023'
    if 'zhao' in stem:                                    return 'zhao_2025_australian_wheat'
    return None


def highlight_targeted(pdf_path: Path, paper_key: str) -> dict:
    cfg = TARGETED[paper_key]
    doc = fitz.open(str(pdf_path))
    n_highlights = 0
    matches_per_phrase = {p: 0 for p in cfg['phrases']}
    page_hits = []
    # Track phrases already highlighted across the entire document so we hit
    # each concept exactly ONCE (the first time it appears).
    already_highlighted = set()

    for page_num, page in enumerate(doc, start=1):
        for phrase in cfg['phrases']:
            if phrase in already_highlighted:
                continue
            quads = page.search_for(phrase, quads=True)
            if not quads:
                continue
            q = quads[0]
            annot = page.add_highlight_annot(q)
            annot.set_colors(stroke=GOLD_RGB)
            annot.set_opacity(0.7)
            annot.set_info(title='WES — methodology adopted',
                           content=f'Cited concept: "{phrase}"')
            annot.update()
            already_highlighted.add(phrase)
            n_highlights += 1
            matches_per_phrase[phrase] += 1
            page_hits.append((page_num, phrase))

    out_path = pdf_path.with_stem(f'{pdf_path.stem}_TARGETED')
    doc.save(str(out_path), garbage=4, deflate=True, clean=True)
    doc.close()

    return {
        'paper_key':   paper_key,
        'description': cfg['description'],
        'input':       pdf_path.name,
        'output':      out_path.name,
        'pages':       page_num,
        'total':       n_highlights,
        'phrases':     matches_per_phrase,
        'page_hits':   page_hits,
    }


def main():
    if len(sys.argv) < 2:
        print('Usage: python highlight_papers_targeted.py <papers_dir>')
        sys.exit(1)

    papers_dir = Path(sys.argv[1])
    pdfs = sorted([
        p for p in papers_dir.glob('*.pdf')
        if not any(p.stem.endswith(s) for s in ('_highlighted','_methodology','_TARGETED'))
    ])

    summary = ['# Targeted Methodology Highlights',
               '',
               'Only the **specific phrases we adopted** from each paper are highlighted '
               'in Boilermaker Gold (#CEB888). Peripheral papers are skipped entirely.',
               '', '---', '']

    for pdf in pdfs:
        key = find_paper_key(pdf.name)
        if key is None:
            print(f'⊝ Skip {pdf.name}  (not in core methodology set)')
            continue
        print(f'→ {pdf.name}')
        try:
            r = highlight_targeted(pdf, key)
            print(f'  ✓ {r["total"]} targeted highlights across {r["pages"]} pages')
            print(f'  → {r["output"]}')
            summary.append(f'## {r["description"]}')
            summary.append(f'- File: `{r["output"]}`')
            summary.append(f'- Highlights: **{r["total"]}**')
            summary.append('')
            summary.append('| Phrase | Hits | Pages |')
            summary.append('|---|---:|---|')
            for ph, n in r['phrases'].items():
                if n == 0: continue
                pages = sorted({p for p, x in r['page_hits'] if x == ph})
                summary.append(f'| `{ph}` | {n} | {", ".join(map(str, pages))} |')
            summary.append('')
        except Exception as e:
            print(f'  ❌ Error: {e}')

    out_md = papers_dir / 'TARGETED_HIGHLIGHTS.md'
    out_md.write_text('\n'.join(summary))
    print(f'\n📝 Summary → {out_md}')


if __name__ == '__main__':
    main()
