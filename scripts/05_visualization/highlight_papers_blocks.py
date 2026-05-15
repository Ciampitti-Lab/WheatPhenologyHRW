"""Highlight whole methodology PARAGRAPHS in reference PDFs.

Different from `highlight_papers.py` (single-keyword): this version uses
PyMuPDF's block-level text extraction to identify and highlight ENTIRE
paragraphs that describe methodology we adopted. A paragraph is highlighted
only if it contains ≥2 anchor concepts from a curated list per paper.

This produces meaningful methodology annotations, not noisy keyword matches.

Usage:
    python scripts/05_visualization/highlight_papers_blocks.py pdfs/
"""
import sys
import re
from pathlib import Path
import fitz   # PyMuPDF


# Boilermaker Gold for annotations (RGB normalized 0..1)
GOLD_RGB = (0.808, 0.722, 0.533)   # #CEB888

# Minimum anchor concepts a block must contain to be highlighted as
# a "methodology paragraph". Higher = more selective.
MIN_ANCHORS_PER_BLOCK = 2

# Minimum block length (chars). Skips short headers / footers / labels.
MIN_BLOCK_CHARS = 80


# ─────────────────────────────────────────────────────────────────────────
# Per-paper ANCHORS — concepts that appear together in methodology paragraphs
# A block needs ≥ MIN_ANCHORS_PER_BLOCK distinct anchors to be highlighted
# ─────────────────────────────────────────────────────────────────────────
PAPER_ANCHORS = {
    'bandaru': {
        'description': 'Bandaru et al. 2020 — PhenoCrop framework',
        'anchors': [
            'accumulated photo-thermal time', 'APTT',
            'Wang and Engel', 'Wang & Engel', 'plant phenology model',
            'Kalman filter', 'data fusion',
            'phenology metrics', 'NDVI phenology',
            'physiological growth stages', 'crop growth stages',
            'temperature response function', 'photoperiod response',
            'green-up', 'end of senescence',
            'reflectance data', 'fused MODIS-Landsat',
            '30 m spatial resolution', '8-day reflectance',
        ],
        'must_include_one_of': [],  # optional gate
    },
    'wang_engel': {
        'description': 'Wang & Engel 1998 — three-phase phenology + beta function',
        'anchors': [
            'beta function', 'temperature response',
            'three phase', 'three-phase', 'three sub-phases',
            'vernalization function', 'photoperiod function',
            'cardinal temperature', 'base temperature',
            'optimum temperature', 'maximum temperature',
            'developmental rate', 'development stage',
            'vegetative phase', 'reproductive phase',
            'thermal time',
            'Tmin', 'Topt', 'Tmax',
            'vernalization days', 'photoperiod sensitivity',
        ],
    },
    'streck': {
        'description': 'Streck et al. 2003 — generalized vernalization function f(V)',
        'anchors': [
            'vernalization function', 'generalized vernalization',
            'vernalization days', 'VD',
            'sigmoidal', 'response function',
            'modified Wang and Engel', 'modified Wang & Engel',
            'cardinal temperature', 'photoperiod function',
            'leaf number', 'final number of leaves',
            'main shoot', 'terminal spikelet',
            'cultivar parameter',
        ],
    },
    'porter_gawith': {
        'description': 'Porter & Gawith 1999 — wheat cardinal temperatures',
        'anchors': [
            'cardinal temperatures',
            'base temperature', 'minimum temperature',
            'optimum temperature', 'maximum temperature',
            'vernalization', 'photoperiod',
            'wheat development', 'phenological stages',
            'temperature limits', 'thermal limits',
            'leaf appearance', 'phyllochron',
            '4.9', '15.7', '1.3',
        ],
    },
    'mcmaster': {
        'description': 'McMaster & Wilhelm 1997 — GDD method 2',
        'anchors': [
            'growing degree days', 'GDD',
            'method 2', 'Method 2',
            'thermal time', 'base temperature',
            'two interpretations', 'one equation',
            'capping', 'upper threshold',
            'Tmin', 'Tmax', 'Tbase',
            'mean daily temperature',
        ],
    },
    'lobert': {
        'description': 'Lobert et al. 2023 — German winter wheat DL benchmark',
        'anchors': [
            'winter wheat', 'phenological stages',
            'temporal U-Net', 'U-Net',
            'Sentinel-1', 'Sentinel-2', 'Landsat 8',
            'BBCH', 'stem elongation', 'harvest',
            'optical and SAR', 'time series',
            'field level', 'absolute error',
            '6 days', 'six days',
            'deep learning', 'random forest',
            'R²', 'R2',
            'Germany', '16,000',
        ],
    },
    'zhao': {
        'description': 'Zhao et al. 2025 — Australian wheat phenology, Sentinel-2',
        'anchors': [
            'wheat and barley', 'wheat',
            'Sentinel-2', 'multi-spectral',
            'GLAI', 'CCC', 'green leaf area index',
            'canopy chlorophyll',
            'double logistic', 'curve fitting',
            'phenology', 'phenological stages',
            'site-specific', 'precision agriculture',
            'Australian', 'grain belt',
            'flag leaf', 'flowering', 'maturity',
            'R²', 'RMSE',
        ],
    },
    'shahhosseini': {
        'description': 'Shahhosseini et al. 2021 — APSIM + ML hybrid yield',
        'anchors': [
            'APSIM', 'crop model', 'crop simulation',
            'machine learning', 'random forest',
            'yield prediction', 'yield estimation',
            'hybrid', 'coupling',
            'extreme climate', 'climate indicators',
            'simulated crop variables', 'feature engineering',
            'Corn Belt',
            'reduce RMSE', 'RMSE',
        ],
    },
}


def detect_paper(filename: str) -> str | None:
    name = Path(filename).stem.lower().replace('_', ' ').replace('-', ' ')
    if 'bandaru' in name or 'phenocrop' in name:    return 'bandaru'
    if 'wang' in name and 'engel' in name:          return 'wang_engel'
    if 'streck' in name and 'vernalization' in name: return 'streck'
    if 'streck' in name:                             return 'streck'
    if 'porter' in name or 'gawith' in name:         return 'porter_gawith'
    if 'mcmaster' in name or 'wilhelm' in name:      return 'mcmaster'
    if 'lobert' in name:                             return 'lobert'
    if 'zhao' in name:                               return 'zhao'
    if 'shahhosseini' in name:                       return 'shahhosseini'
    return None


def count_anchor_hits(text_lower: str, anchors: list[str]) -> tuple[int, list[str]]:
    """Count distinct anchor concepts present in text. Returns (count, matches)."""
    found = set()
    for a in anchors:
        if a.lower() in text_lower:
            # Normalize duplicate anchors (e.g., 'Wang and Engel' / 'Wang & Engel')
            key = re.sub(r'[&]', 'and', a.lower())
            found.add(key)
    return len(found), sorted(found)


def highlight_pdf_blocks(pdf_path: Path, paper_key: str) -> dict:
    cfg = PAPER_ANCHORS[paper_key]
    anchors = cfg['anchors']
    doc = fitz.open(str(pdf_path))

    n_blocks_highlighted = 0
    matched_blocks: list[dict] = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text('blocks')
        # blocks tuple: (x0, y0, x1, y1, text, block_no, block_type)
        for b in blocks:
            x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
            text = b[4] if len(b) > 4 else ''
            block_type = b[6] if len(b) > 6 else 0
            if block_type != 0:                  # 0 = text, 1 = image
                continue
            if len(text) < MIN_BLOCK_CHARS:
                continue

            n_anchors, found = count_anchor_hits(text.lower(), anchors)
            if n_anchors < MIN_ANCHORS_PER_BLOCK:
                continue

            # Highlight by re-finding each anchor's exact quad in the page
            # (this gives line-accurate, multi-line-aware highlights)
            block_rect = fitz.Rect(x0, y0, x1, y1)

            # Get text rectangles within this block to highlight precisely
            highlight_quads = []
            for a in anchors:
                if a.lower() not in text.lower():
                    continue
                # Find all rectangles for this anchor in the page
                quads = page.search_for(a, quads=True, clip=block_rect)
                highlight_quads.extend(quads)

            # Also highlight the full block by adding a single rectangle annot
            # (this draws a soft fill across the whole paragraph)
            annot = page.add_highlight_annot(block_rect)
            annot.set_colors(stroke=GOLD_RGB)
            annot.set_opacity(0.20)              # gentle paragraph wash
            annot.set_info(title='Methodology adopted',
                           content=f'Anchors: {", ".join(found)}')
            annot.update()

            # Stronger highlights on the anchor phrases themselves
            for q in highlight_quads:
                a2 = page.add_highlight_annot(q)
                a2.set_colors(stroke=GOLD_RGB)
                a2.set_opacity(0.55)
                a2.update()

            n_blocks_highlighted += 1
            matched_blocks.append({
                'page': page_num,
                'snippet': text[:200].replace('\n', ' ').strip(),
                'anchors_found': found,
                'n_anchors': n_anchors,
            })

    out_path = pdf_path.with_stem(f'{pdf_path.stem}_methodology')
    doc.save(str(out_path), garbage=4, deflate=True, clean=True)
    doc.close()

    return {
        'paper_key':     paper_key,
        'description':   cfg['description'],
        'input':         str(pdf_path.name),
        'output':        str(out_path.name),
        'pages':         page_num,
        'n_blocks':      n_blocks_highlighted,
        'matched_blocks': matched_blocks,
    }


def main():
    if len(sys.argv) < 2:
        print('Usage: python highlight_papers_blocks.py <papers_dir>')
        sys.exit(1)

    papers_dir = Path(sys.argv[1])
    pdfs = sorted([
        p for p in papers_dir.glob('*.pdf')
        if not p.stem.endswith('_highlighted') and not p.stem.endswith('_methodology')
    ])
    if not pdfs:
        print(f'❌ No source PDFs in {papers_dir}/')
        sys.exit(1)

    print(f'Found {len(pdfs)} PDFs in {papers_dir}/')
    print(f'Threshold: ≥{MIN_ANCHORS_PER_BLOCK} anchor concepts per block, ≥{MIN_BLOCK_CHARS} chars\n')

    summary_lines = [
        '# Methodology Paragraphs Highlighted',
        '',
        'Block-level annotations: each highlighted *paragraph* contains '
        f'≥{MIN_ANCHORS_PER_BLOCK} concepts that connect to the WES framework. '
        'Boilermaker Gold (#CEB888) wash on the full paragraph; brighter highlight '
        'on the anchor phrases themselves.',
        '',
        '---',
        '',
    ]

    for pdf in pdfs:
        key = detect_paper(pdf.name)
        if key is None:
            print(f'⚠ Skip {pdf.name} (no paper match)')
            continue

        print(f'→ {pdf.name}  ({key})')
        try:
            result = highlight_pdf_blocks(pdf, key)
            print(f'  ✓ {result["n_blocks"]} methodology paragraphs across {result["pages"]} pages')
            print(f'  → {result["output"]}')

            summary_lines.append(f'## {result["description"]}')
            summary_lines.append(f'- Input:  `{result["input"]}`')
            summary_lines.append(f'- Output: `{result["output"]}`')
            summary_lines.append(f'- Methodology paragraphs highlighted: **{result["n_blocks"]}** '
                                 f'(of {result["pages"]} pages)')
            summary_lines.append('')
            if result['matched_blocks']:
                summary_lines.append('### Sample passages')
                summary_lines.append('')
                for mb in result['matched_blocks'][:6]:    # first 6
                    summary_lines.append(f'**p.{mb["page"]}** — *anchors: {", ".join(mb["anchors_found"])}*')
                    summary_lines.append('')
                    summary_lines.append(f'> {mb["snippet"]}')
                    summary_lines.append('')
                if len(result['matched_blocks']) > 6:
                    summary_lines.append(f'*...and {len(result["matched_blocks"])-6} more.*')
                    summary_lines.append('')
            summary_lines.append('')
        except Exception as e:
            print(f'  ❌ Error: {e}')

    summary_path = papers_dir / 'METHODOLOGY_SUMMARY.md'
    summary_path.write_text('\n'.join(summary_lines))
    print(f'\n📝 Methodology summary → {summary_path}')


if __name__ == '__main__':
    main()
