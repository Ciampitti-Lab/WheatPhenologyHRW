"""Auto-highlight key passages in reference PDFs that informed our WES methodology.

For each paper, the script searches for specific phrases that we built upon and
adds yellow/gold highlight annotations. Output PDFs are saved alongside the input
with `_highlighted` suffix.

Usage:
    # 1. Place reference PDFs in papers/
    # 2. Run:
    python scripts/05_visualization/highlight_papers.py papers/

Output: each PDF gets a `<name>_highlighted.pdf` in the same folder, plus a
`papers/HIGHLIGHTS_SUMMARY.md` listing every match found per paper.
"""
import sys
import re
from pathlib import Path
import fitz   # PyMuPDF


# Boilermaker Gold for annotations (RGB normalized to 0..1)
GOLD_RGB = (0.808, 0.722, 0.533)   # #CEB888

# ─────────────────────────────────────────────────────────────────────────
# Per-paper search terms — passages we adopted into WES
# ─────────────────────────────────────────────────────────────────────────
PAPER_KEYWORDS = {
    'bandaru': {
        'description': 'Bandaru et al. 2020 — PhenoCrop (foundation framework)',
        'phrases': [
            'accumulated photo-thermal time',
            'APTT',
            'Wang and Engel',
            'Wang & Engel',
            'temperature response function',
            'photoperiod response',
            'physiological growth stages',
            'Kalman Filter',
            'NDVI phenology',
            'green-up',
            'end of senescence',
            'fused MODIS-Landsat',
            'corn and soybean',
            'phenology metrics',
            'crop growth stages',
        ],
    },
    'wang_engel': {
        'description': 'Wang & Engel 1998 — three-phase phenology model + beta function',
        'phrases': [
            'three-phase',
            'beta function',
            'cardinal temperature',
            'optimum temperature',
            'minimum temperature',
            'maximum temperature',
            'vernalization',
            'photoperiod',
            'development stage',
            'developmental rate',
            'temperature response',
            'thermal time',
            'Tmin', 'Topt', 'Tmax',
            'reproductive phase',
            'vegetative phase',
        ],
    },
    'streck': {
        'description': 'Streck et al. 2003 — generalized vernalization function f(V)',
        'phrases': [
            'vernalization days',
            'VD',
            'sigmoidal',
            'generalized vernalization',
            'modified Wang and Engel',
            'modified Wang & Engel',
            'cardinal temperatures for vernalization',
            'f(V)',
            'photoperiod function',
            'cultivar parameter',
            'leaf number',
            'final number of leaves',
            'main shoot',
            'terminal spikelet',
        ],
    },
    'porter_gawith': {
        'description': 'Porter & Gawith 1999 — wheat cardinal temperatures review',
        'phrases': [
            'cardinal temperatures',
            'base temperature',
            'optimum temperature',
            'maximum temperature',
            'vernalization',
            'photoperiod',
            'wheat development',
            'phenological stages',
            'review',
            'thermal limits',
            '4.9',  # often cited as Topt for vernalization
            '15.7', # Tmax for vernalization
            '1.3',  # Tmin for vernalization
        ],
    },
    'mcmaster': {
        'description': 'McMaster & Wilhelm 1997 — GDD method 2',
        'phrases': [
            'growing degree days',
            'GDD',
            'method 2',
            'thermal time',
            'base temperature',
            'two interpretations',
            'one equation',
            'Tmin', 'Tmax', 'Tbase',
            'capping',
        ],
    },
    'lobert': {
        'description': 'Lobert et al. 2023 — German winter wheat phenology DL benchmark',
        'phrases': [
            'winter wheat phenology',
            'temporal U-Net',
            'U-Net',
            'Sentinel-1',
            'Sentinel-2',
            'Landsat 8',
            'phenological stages',
            'BBCH',
            'stem elongation',
            'harvest',
            'optical and SAR',
            'field level',
            '16,000 fields',
            'Germany',
            'absolute error',
            'six days',
            '50.1', '65.5',
            'R²',
            'deep learning',
        ],
    },
    'zhao': {
        'description': 'Zhao et al. 2025 — Australian wheat phenology, Sentinel-2',
        'phrases': [
            'wheat and barley',
            'phenology',
            'Sentinel-2',
            'GLAI',
            'CCC',
            'canopy chlorophyll',
            'green leaf area',
            'double logistic',
            'site-specific',
            'precision agriculture',
            'Australian',
            'flag leaf',
            'flowering',
            'maturity',
            'R² = 0.7',
            'RMSE',
            '7.66',
        ],
    },
    'shahhosseini': {
        'description': 'Shahhosseini et al. 2021 — APSIM + ML hybrid (US Corn Belt)',
        'phrases': [
            'APSIM',
            'machine learning',
            'crop model',
            'random forest',
            'yield prediction',
            'hybrid',
            'crop simulation',
            'extreme climate',
            'feature engineering',
            'simulated crop variables',
            'Corn Belt',
            'reduce RMSE',
        ],
    },
    'phenocrop': {  # alias for bandaru in case file is named phenocrop.pdf
        'description': 'PhenoCrop framework (Bandaru et al. 2020)',
        'phrases': [
            'accumulated photo-thermal time',
            'APTT',
            'Wang and Engel',
            'NDVI phenology',
            'phenology metrics',
            'corn and soybean',
            'physiological growth stages',
        ],
    },
}


def detect_paper(filename: str) -> str | None:
    """Match filename to a paper key in PAPER_KEYWORDS."""
    name = Path(filename).stem.lower().replace('_', ' ').replace('-', ' ')
    for key in PAPER_KEYWORDS:
        if key in name or key.replace('_', '') in name:
            return key
    # Heuristics
    if 'bandaru' in name or 'phenocrop' in name:    return 'bandaru'
    if 'wang' in name and 'engel' in name:          return 'wang_engel'
    if 'streck' in name:                             return 'streck'
    if 'porter' in name or 'gawith' in name:         return 'porter_gawith'
    if 'mcmaster' in name or 'wilhelm' in name:      return 'mcmaster'
    if 'lobert' in name:                             return 'lobert'
    if 'zhao' in name:                               return 'zhao'
    if 'shahhosseini' in name:                       return 'shahhosseini'
    return None


def highlight_pdf(pdf_path: Path, paper_key: str) -> dict:
    """Highlight all occurrences of paper_key's phrases in the PDF."""
    cfg = PAPER_KEYWORDS[paper_key]
    phrases = cfg['phrases']

    doc = fitz.open(str(pdf_path))
    matches = {p: 0 for p in phrases}
    total_highlights = 0

    for page_num, page in enumerate(doc, start=1):
        for phrase in phrases:
            # Case-insensitive search; quads cover multi-line matches
            instances = page.search_for(phrase, quads=True)
            for q in instances:
                annot = page.add_highlight_annot(q)
                annot.set_colors(stroke=GOLD_RGB)
                annot.set_info(title='WES methodology', content=f'Adopted: {phrase}')
                annot.update()
                matches[phrase] += 1
                total_highlights += 1

    out_path = pdf_path.with_stem(f'{pdf_path.stem}_highlighted')
    doc.save(str(out_path), garbage=4, deflate=True, clean=True)
    doc.close()

    return {
        'paper_key':      paper_key,
        'description':    cfg['description'],
        'input':          str(pdf_path.name),
        'output':         str(out_path.name),
        'total':          total_highlights,
        'per_phrase':     matches,
        'pages':          page_num,
    }


def main():
    if len(sys.argv) < 2:
        print('Usage: python highlight_papers.py <papers_dir>')
        sys.exit(1)

    papers_dir = Path(sys.argv[1])
    if not papers_dir.exists():
        print(f'❌ Directory not found: {papers_dir}')
        print(f'Create it with:  mkdir -p {papers_dir}')
        sys.exit(1)

    pdfs = sorted(papers_dir.glob('*.pdf'))
    pdfs = [p for p in pdfs if not p.stem.endswith('_highlighted')]
    if not pdfs:
        print(f'❌ No PDFs in {papers_dir}/')
        print('Place reference papers there with descriptive names:')
        for k, v in PAPER_KEYWORDS.items():
            print(f'  {k}.pdf   — {v["description"]}')
        sys.exit(1)

    print(f'Found {len(pdfs)} PDFs in {papers_dir}/')
    summary_lines = ['# Reference Papers — Highlighted Passages', '',
                     f'Auto-generated by `highlight_papers.py`. Phrases used in our WES '
                     f'methodology are highlighted in **Boilermaker Gold (#CEB888)** within each '
                     f'output PDF. Matches per paper listed below.', '', '---', '']

    for pdf in pdfs:
        key = detect_paper(pdf.name)
        if key is None:
            print(f'⚠ Could not match {pdf.name} to a known paper — skipping.')
            print(f'   Rename it to one of: {", ".join(PAPER_KEYWORDS.keys())}.pdf')
            continue
        print(f'\n→ Highlighting {pdf.name}  (paper: {key})')
        try:
            result = highlight_pdf(pdf, key)
            print(f'  ✓ {result["total"]} highlights across {result["pages"]} pages')
            print(f'  → saved: {result["output"]}')

            summary_lines.append(f'## {result["description"]}')
            summary_lines.append(f'- Input file:  `{result["input"]}`')
            summary_lines.append(f'- Output file: `{result["output"]}`')
            summary_lines.append(f'- Pages: {result["pages"]}, total highlights: **{result["total"]}**')
            summary_lines.append('')
            summary_lines.append('| Phrase | Hits |')
            summary_lines.append('|---|---:|')
            for p, n in sorted(result['per_phrase'].items(), key=lambda x: -x[1]):
                if n > 0:
                    summary_lines.append(f'| {p} | {n} |')
            summary_lines.append('')
        except Exception as e:
            print(f'  ❌ Error: {e}')

    summary_path = papers_dir / 'HIGHLIGHTS_SUMMARY.md'
    summary_path.write_text('\n'.join(summary_lines))
    print(f'\n📝 Summary saved → {summary_path}')


if __name__ == '__main__':
    main()
