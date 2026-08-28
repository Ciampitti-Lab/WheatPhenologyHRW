"""Poll the R13a Earth Engine export tasks."""
import sys
import ee
ee.Initialize(project='propane-primacy-481403-u3')
want = {'bufsens_L30_raw', 'bufsens_L30_cdl', 'bufsens_S30_raw', 'bufsens_S30_cdl'}
seen, done = {}, 0
for t in ee.data.listOperations():
    md = t.get('metadata', {})
    d = md.get('description', '')
    if d in want and d not in seen:
        seen[d] = md.get('state', '?')
for d in sorted(want):
    st = seen.get(d, 'NOT FOUND')
    print(f'{d:20s} {st}')
    if st in ('SUCCEEDED', 'FAILED', 'CANCELLED'):
        done += 1
sys.exit(0 if done == len(want) else 1)
