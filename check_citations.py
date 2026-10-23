import re, glob
bib = open('refs_updated.bib', encoding='utf-8').read()
entries = re.findall(r'@(\w+)\{([^,]+),(.*?)\n\}', bib, re.S)
titles = {}
for typ, key, body in entries:
    t = re.search(r'title\s*=\s*[{\"\'](.+?)[}\"\'],?\s*\n', body, re.S)
    titles[key] = (typ, (t.group(1).strip()[:72] if t else '?'))
cited = set()
pat = re.compile(r'cite\{([^}]+)\}')
for f in glob.glob('*.tex'):
    for m in pat.findall(open(f, encoding='utf-8', errors='ignore').read()):
        for k in m.split(','):
            cited.add(k.strip())
print('total bib:', len(titles), '| cited:', len(cited))
print()
print('=== UNCITED ===')
for k, (typ, t) in titles.items():
    if k not in cited:
        print('%-16s %-12s %s' % (k, typ, t))
