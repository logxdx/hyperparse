import re, sys, json
src = open(sys.argv[1]).read()
body = src[src.index('static AGL_SUBSET'):]
body = body[body.index('['):body.index('];')]
ents = re.findall(r'\(\s*"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*\)', body)
def unesc(v):  # rust escapes: \u{XXXX}, \\, \", \'
    return re.sub(r'\\(u\{[0-9A-Fa-f]+\}|.)',
                  lambda m: chr(int(m.group(1)[2:-1], 16)) if m.group(1).startswith('u{') else m.group(1), v)
ents = [(n, unesc(v)) for n, v in ents]
assert [n for n, _ in ents] == sorted(n for n, _ in ents), "not sorted"
out = ['// Generated from liteparse crates/liteparse/src/glyph_names.rs AGL_SUBSET (Apache-2.0).',
       '// Sorted by name for binary search. Do not edit; regenerate with tools/gen_agl.py.',
       '#pragma once', '#include <string_view>', 'namespace hp::agl {',
       'struct Entry { std::string_view name, text; };',
       f'inline constexpr Entry kTable[{len(ents)}] = {{']
for n, v in ents:
    out.append('  {%s, %s},' % (json.dumps(n), json.dumps(v, ensure_ascii=False)))
out += ['};', '}  // namespace hp::agl', '']
open(sys.argv[2], 'w').write('\n'.join(out))
print(len(ents), 'entries; sample', ents[:2], ents[-2:])
