"""Line-level methodology highlighter — clean marker-style highlights of
the specific sentences/clauses that describe the methodology we adopted.

No paragraph "bubbles" — only the actual lines of text get highlighted,
following the natural shape of the text (multi-line phrases included).

For each core paper, 2-3 specific sentences/clauses are highlighted —
the exact lines that explain what we took from that paper.

Usage:
    python scripts/05_visualization/highlight_papers_lines.py pdfs/
"""
import sys
from pathlib import Path
import fitz   # PyMuPDF


GOLD_RGB = (0.808, 0.722, 0.533)   # #CEB888


# ─────────────────────────────────────────────────────────────────────────
# 2-3 SPECIFIC sentences/clauses per paper — the exact lines explaining
# what we adopted. Each phrase should be 5-25 words and unique enough to
# match exactly once.
# ─────────────────────────────────────────────────────────────────────────
LINES = {
    'bandaru_2020_phenocrop': {
        'description': 'Bandaru 2020 — PhenoCrop framework',
        'lines': [
            ('PhenoCrop framework constitutes three components',
             '3-component architecture (KFR + RS-phenology + APTT) — we extend it with f(V)'),
            ('built using a beta function with three',
             'The exact f(T) construction (beta function with Tmin/Topt/Tmax) we use'),
            ('daily photo-thermal time',
             'The APTT accumulation principle — we extend with vernalization to WES'),
        ],
    },
    'wang_engel_1998': {
        'description': 'Wang & Engel 1998 — phenology model with f(T)·f(V)·f(P)',
        'lines': [
            ('A wheat phenology model based on the effects of temperature',
             'The foundational dDVS/dt = R_max · f(T) · f(V) · f(P) equation'),
            ('vernalization function is given',
             'Their f(V) (linear three-stage) — Streck 2003 later generalized this to sigmoidal'),
        ],
    },
    'streck_2003_vernalization': {
        'description': 'Streck 2003 — generalized f(V) = VD^5 / (22.5^5 + VD^5)',
        'lines': [
            ('22.5 (half of the maximum re-',
             'The 22.5-day half-saturation constant in the f(V) sigmoidal — exact value we use'),
            ('1.3, 4.9, and 15.7',
             'Vernalization cardinal temperatures (Tmin, Topt, Tmax) we use in our simulator'),
            ('VD was calculated with Eq. [5]',
             'The recipe for accumulating vernalization days — directly implemented in thermal.py'),
        ],
    },
    'porter_gawith_1999': {
        'description': 'Porter & Gawith 1999 — wheat cardinal temperatures (source values)',
        'lines': [
            ('Vernalization Tmin',
             'Source of our vernalization Tmin = -1.3 °C value'),
            ('Topt 4.9',
             'Source of our vernalization Topt = 4.9 °C value'),
            ('Tmax 15.7',
             'Source of our vernalization Tmax = 15.7 °C value'),
        ],
    },
    'mcmaster_wilhelm_1997': {
        'description': 'McMaster & Wilhelm 1997 — GDD Method 2',
        'lines': [
            ('Method 1 accumulates fewer GDD than Method 2',
             'The Method 1 vs Method 2 distinction — we use Method 2 throughout'),
            ('incorporating an upper threshold',
             'The Tmax capping rule that defines Method 2 — we cap at Topt per phase'),
        ],
    },
    'lobert_2023': {
        'description': 'Lobert 2023 — DL benchmark we compare against',
        'lines': [
            ('temporal U-Net',
             'Their architecture — contrast with our linear + tree-ensemble approach'),
            ('absolute error of less than six days',
             'Their headline ±6 day claim — we report ±10 day with ≥94% on 4 critical stages'),
        ],
    },
    'zhao_2025_australian_wheat': {
        'description': 'Zhao 2025 — Australian SOTA we compare against',
        'lines': [
            ('main development stages of cereal growth',
             'Their target stages — the same set we predict but in U.S. Plains'),
            ('flag leaf',
             'Their flag-leaf R²=0.70 vs our R²=0.82 — direct head-to-head'),
        ],
    },
}


def detect_paper(filename: str) -> str | None:
    stem = Path(filename).stem.lower()
    for k in LINES:
        if k in stem:
            return k
    if 'bandaru' in stem and 'phenocrop' in stem:    return 'bandaru_2020_phenocrop'
    if 'wang' in stem and 'engel' in stem:            return 'wang_engel_1998'
    if 'streck' in stem and 'vernalization' in stem: return 'streck_2003_vernalization'
    if 'porter' in stem or 'gawith' in stem:         return 'porter_gawith_1999'
    if 'mcmaster' in stem or 'wilhelm' in stem:      return 'mcmaster_wilhelm_1997'
    if 'lobert' in stem:                              return 'lobert_2023'
    if 'zhao' in stem:                                return 'zhao_2025_australian_wheat'
    return None


def highlight_lines(pdf_path: Path, paper_key: str) -> dict:
    cfg = LINES[paper_key]
    doc = fitz.open(str(pdf_path))
    flags = fitz.TEXT_DEHYPHENATE | fitz.TEXT_PRESERVE_WHITESPACE
    used = []

    for phrase, why in cfg['lines']:
        # Search through all pages for first occurrence
        target_page = None
        target_quads = None
        for page_num, page in enumerate(doc, start=1):
            quads = page.search_for(phrase, quads=True, flags=flags)
            if quads:
                target_page = page
                target_page_num = page_num
                target_quads = quads
                break

        if target_quads is None:
            used.append({'phrase': phrase, 'why': why,
                         'page': None, 'status': 'not found'})
            continue

        # Highlight ONLY the line-level quads (no paragraph wash).
        # `add_highlight_annot` accepts a list of quads → one annotation
        # following the natural shape of the text (multi-line aware).
        annot = target_page.add_highlight_annot(target_quads)
        annot.set_colors(stroke=GOLD_RGB)
        annot.set_opacity(0.55)
        annot.set_info(title='WES — adopted', content=f'WHY: {why}')
        annot.update()

        used.append({'phrase': phrase, 'why': why,
                     'page': target_page_num, 'status': 'highlighted'})

    out_path = pdf_path.with_stem(f'{pdf_path.stem}_LINES')
    doc.save(str(out_path), garbage=4, deflate=True, clean=True)
    doc.close()
    return {
        'paper_key':   paper_key,
        'description': cfg['description'],
        'output':      out_path.name,
        'lines':       used,
    }


def main():
    if len(sys.argv) < 2:
        print('Usage: python highlight_papers_lines.py <papers_dir>')
        sys.exit(1)
    papers_dir = Path(sys.argv[1])
    pdfs = sorted([p for p in papers_dir.glob('*.pdf')
                   if not any(p.stem.endswith(s) for s in
                              ('_highlighted','_methodology','_TARGETED','_PASSAGES','_LINES'))])

    summary = ['# Methodology highlights (line-level)', '',
               'Specific sentences/clauses we adopted from each paper, '
               'highlighted in **Boilermaker Gold (#CEB888)** — marker-style, '
               'no paragraph bubbles.', '', '---', '']

    for pdf in pdfs:
        key = detect_paper(pdf.name)
        if key is None:
            continue
        print(f'→ {pdf.name}')
        try:
            r = highlight_lines(pdf, key)
            ok = sum(1 for ln in r['lines'] if ln['status']=='highlighted')
            print(f'  ✓ {ok}/{len(r["lines"])} lines highlighted → {r["output"]}')

            summary.append(f'## {r["description"]}')
            summary.append(f'`{r["output"]}`')
            summary.append('')
            summary.append('| WHY we cite | Highlighted phrase | Page |')
            summary.append('|---|---|---:|')
            for ln in r['lines']:
                pg = ln['page'] if ln['page'] else '⚠ not found'
                summary.append(f'| {ln["why"]} | *{ln["phrase"]}* | {pg} |')
            summary.append('')
        except Exception as e:
            print(f'  ❌ {e}')

    out_md = papers_dir / 'METHODOLOGY_LINES.md'
    out_md.write_text('\n'.join(summary))
    print(f'\n📝 Summary → {out_md}')


if __name__ == '__main__':
    main()
