"""Passage-level methodology highlighter (final version).

For each core paper, 1-3 PASSAGES (full paragraphs or contiguous text blocks)
that explain the methodology we adopted are highlighted in Boilermaker Gold.
Each passage is anchored by a unique short phrase; the script finds that
anchor, identifies the paragraph block containing it, and highlights the
entire block.

Usage:
    python scripts/05_visualization/highlight_papers_passages.py pdfs/
"""
import sys
from pathlib import Path
import fitz   # PyMuPDF


GOLD_RGB = (0.808, 0.722, 0.533)   # #CEB888


# ─────────────────────────────────────────────────────────────────────────
# Per-paper anchored passages — 1-3 per paper.
# `anchor` is a unique 5-15 word phrase that exists verbatim inside the
# paragraph we want to highlight. The script highlights the FULL paragraph
# block containing that anchor.
# ─────────────────────────────────────────────────────────────────────────
PASSAGES = {
    'bandaru_2020_phenocrop': {
        'description': 'Bandaru 2020 — PhenoCrop framework (foundation we extend)',
        'passages': [
            {'anchor': 'PhenoCrop framework constitutes three components',
             'why': 'The three-component architecture we extend with vernalization'},
            {'anchor': 'accumulated photo-thermal time',
             'why': 'APTT definition — the variable we extend to WES'},
            {'anchor': 'plant phenology model',
             'why': 'Their use of Wang-Engel as the conversion mechanism'},
        ],
    },
    'wang_engel_1998': {
        'description': 'Wang & Engel 1998 — temperature/vernalization/photoperiod functions',
        'passages': [
            {'anchor': 'beta',
             'why': 'The temperature response function f(T) we use'},
            {'anchor': 'vernalization function',
             'why': 'The original vernalization formulation later modified by Streck (2003)'},
            {'anchor': 'photoperiod',
             'why': 'The photoperiod response f(P) we use during phase 2'},
        ],
    },
    'streck_2003_vernalization': {
        'description': 'Streck 2003 — generalized vernalization function f(V) = VD^5 / (22.5^5 + VD^5)',
        'passages': [
            {'anchor': 'generalized vernalization response',
             'why': 'The exact f(V) sigmoidal formulation we adopt in WES'},
            {'anchor': 'cardinal temperatures',
             'why': 'Vernalization cardinal temperatures (1.3 / 4.9 / 15.7 °C)'},
        ],
    },
    'porter_gawith_1999': {
        'description': 'Porter & Gawith 1999 — wheat cardinal temperatures review',
        'passages': [
            {'anchor': 'cardinal temperatures',
             'why': 'Source of the cardinal temperature values we use'},
            {'anchor': 'vernalization',
             'why': 'Vernalization temperature limits we adopt'},
        ],
    },
    'mcmaster_wilhelm_1997': {
        'description': 'McMaster & Wilhelm 1997 — GDD Method 2',
        'passages': [
            {'anchor': 'Method 2',
             'why': 'The exact GDD calculation rule we use (with capping)'},
            {'anchor': 'upper threshold',
             'why': 'The Tmax capping rule that distinguishes Method 2'},
        ],
    },
    'lobert_2023': {
        'description': 'Lobert 2023 — German DL benchmark we compare against',
        'passages': [
            {'anchor': 'temporal U-Net',
             'why': 'The deep learning architecture we benchmark against'},
            {'anchor': 'absolute error of less than six days',
             'why': 'The headline accuracy claim we compare with our ±10-day metric'},
        ],
    },
    'zhao_2025_australian_wheat': {
        'description': 'Zhao 2025 — Australian SOTA we compare against',
        'passages': [
            {'anchor': 'leaf area index',
             'why': 'Their canopy retrieval approach (alternative to our HLS NDVI)'},
            {'anchor': 'flag leaf',
             'why': 'Their reported R² for flag leaf — direct comparison anchor'},
        ],
    },
}


def detect_paper(filename: str) -> str | None:
    stem = Path(filename).stem.lower()
    for k in PASSAGES:
        if k in stem:
            return k
    if 'bandaru' in stem and 'phenocrop' in stem: return 'bandaru_2020_phenocrop'
    if 'wang' in stem and 'engel' in stem:        return 'wang_engel_1998'
    if 'streck' in stem and 'vernalization' in stem: return 'streck_2003_vernalization'
    if 'porter' in stem or 'gawith' in stem:       return 'porter_gawith_1999'
    if 'mcmaster' in stem or 'wilhelm' in stem:    return 'mcmaster_wilhelm_1997'
    if 'lobert' in stem:                            return 'lobert_2023'
    if 'zhao' in stem:                              return 'zhao_2025_australian_wheat'
    return None


def find_block_containing(page: fitz.Page, anchor: str) -> tuple[fitz.Rect, str] | None:
    """Find the text block (paragraph) on this page that contains the anchor phrase.
    Uses TEXT_DEHYPHENATE flag so phrases broken across lines are still found.
    """
    flags = fitz.TEXT_DEHYPHENATE | fitz.TEXT_PRESERVE_WHITESPACE
    quads = page.search_for(anchor, quads=True, flags=flags)
    if not quads:
        return None
    # Use the first match's bounding box center
    q = quads[0]
    if hasattr(q, 'rect'):
        anchor_rect = q.rect
    else:
        anchor_rect = fitz.Rect(q[0], q[1], q[2], q[3]) if isinstance(q, (list, tuple)) else q

    cx = (anchor_rect.x0 + anchor_rect.x1) / 2
    cy = (anchor_rect.y0 + anchor_rect.y1) / 2

    # Get all text blocks (paragraphs) on the page
    blocks = page.get_text('blocks')
    for b in blocks:
        x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
        text = b[4] if len(b) > 4 else ''
        block_type = b[6] if len(b) > 6 else 0
        if block_type != 0:                  # skip image blocks
            continue
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            return fitz.Rect(x0, y0, x1, y1), text.strip()
    # Fallback: use the anchor's own rect
    return anchor_rect, anchor


def highlight_passages(pdf_path: Path, paper_key: str) -> dict:
    cfg = PASSAGES[paper_key]
    doc = fitz.open(str(pdf_path))
    used_passages = []
    seen_blocks: set[tuple[int, float, float]] = set()   # avoid duplicate blocks across passages

    for passage in cfg['passages']:
        anchor = passage['anchor']
        why = passage['why']
        # Find first page containing the anchor
        target = None
        for page_num, page in enumerate(doc, start=1):
            r = find_block_containing(page, anchor)
            if r is None:
                continue
            block_rect, block_text = r
            block_key = (page_num, round(block_rect.x0), round(block_rect.y0))
            if block_key in seen_blocks:
                continue            # already highlighted via another anchor
            seen_blocks.add(block_key)
            target = (page_num, page, block_rect, block_text)
            break

        if target is None:
            used_passages.append({'anchor': anchor, 'why': why,
                                  'page': None, 'snippet': None,
                                  'status': 'not found'})
            continue

        page_num, page, block_rect, block_text = target
        # 1) Soft wash on the entire paragraph
        a1 = page.add_highlight_annot(block_rect)
        a1.set_colors(stroke=GOLD_RGB)
        a1.set_opacity(0.25)
        a1.set_info(title=f'WES methodology', content=f'WHY: {why}')
        a1.update()
        # 2) Stronger underline on the anchor phrase itself
        for q in page.search_for(anchor, quads=True, clip=block_rect):
            a2 = page.add_highlight_annot(q)
            a2.set_colors(stroke=GOLD_RGB)
            a2.set_opacity(0.7)
            a2.update()

        used_passages.append({
            'anchor':  anchor,
            'why':     why,
            'page':    page_num,
            'snippet': block_text[:280].replace('\n', ' ').strip(),
            'status':  'highlighted',
        })

    out_path = pdf_path.with_stem(f'{pdf_path.stem}_PASSAGES')
    doc.save(str(out_path), garbage=4, deflate=True, clean=True)
    doc.close()

    return {
        'paper_key':   paper_key,
        'description': cfg['description'],
        'input':       pdf_path.name,
        'output':      out_path.name,
        'passages':    used_passages,
    }


def main():
    if len(sys.argv) < 2:
        print('Usage: python highlight_papers_passages.py <papers_dir>')
        sys.exit(1)
    papers_dir = Path(sys.argv[1])
    pdfs = sorted([p for p in papers_dir.glob('*.pdf')
                   if not any(p.stem.endswith(s) for s in
                              ('_highlighted', '_methodology', '_TARGETED', '_PASSAGES'))])

    summary = ['# Methodology Passages — what we adopted from each paper',
               '',
               'For each core paper, the **1-3 paragraphs that explain the methodology '
               'we adopted** are highlighted in Boilermaker Gold (#CEB888). Below is the '
               'snippet of each highlighted passage with WHY we cite it.',
               '', '---', '']

    for pdf in pdfs:
        key = detect_paper(pdf.name)
        if key is None:
            continue
        print(f'→ {pdf.name}  ({key})')
        try:
            r = highlight_passages(pdf, key)
            ok = sum(1 for p in r['passages'] if p['status']=='highlighted')
            total = len(r['passages'])
            print(f'  ✓ {ok}/{total} passages highlighted → {r["output"]}')
            summary.append(f'## {r["description"]}')
            summary.append(f'`{r["output"]}`')
            summary.append('')
            for p in r['passages']:
                if p['status'] == 'highlighted':
                    summary.append(f'### Passage on page {p["page"]} — *{p["why"]}*')
                    summary.append('')
                    summary.append(f'> {p["snippet"]}')
                    summary.append('')
                else:
                    summary.append(f'### *{p["why"]}* — ⚠ anchor "{p["anchor"]}" not found verbatim')
                    summary.append('')
            summary.append('---')
            summary.append('')
        except Exception as e:
            print(f'  ❌ {e}')

    out_md = papers_dir / 'METHODOLOGY_PASSAGES.md'
    out_md.write_text('\n'.join(summary))
    print(f'\n📝 Summary → {out_md}')


if __name__ == '__main__':
    main()
