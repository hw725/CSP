#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
toc_dashboard.py — 목차 startparagraph 시각 검증 대시보드 (HTML) 생성

각 서명의 목차 항목별로 startparagraph(N) 이 교정(정보)보강문서의 ##N 문단을
실제로 가리키는 내용(원문·번역문)을 나란히 보여준다. 사람이 눈으로
"제목이 맞는지 / 작품 시작이 맞는지" 확인하는 용도.

대상(두 레이아웃 자동 탐색):
  1) {PC_ROOT}/{책}/{책}_목차.xml + {책}_교정정보보강문서(완료).txt
  2) {PC_ROOT}/병렬말뭉치 결과물/{책}_결과물/{책}_목차.xml + {책}_교정·정보보강문서.txt

출력: {OUT}/index.html + {OUT}/{책}.html  (한 폴더)

사용:
  py toc_dashboard.py --pc-root "...\\PC2025(xlsx)" --out "...\\목차_startparagraph_검증"
"""

import argparse
import html
import re
from pathlib import Path
from typing import List, Optional, Tuple

# ---- 정규화 ----
_HAN = re.compile(r'[㐀-䶿一-鿿]')
_YEAR = re.compile(r'^(元年|[一二三四五六七八九十百千]+年)')


def han(s: str) -> str:
    return ''.join(_HAN.findall(s or ''))


def strip_num(s: str) -> str:
    return re.sub(r'^\s*\d+\s*[.．、]\s*', '', s or '').strip()


# ---- 파일 파싱 ----

def parse_toc(toc_path: Path) -> List[dict]:
    txt = toc_path.read_text(encoding='utf-8-sig')
    out = []
    for vm in re.finditer(r'<volume name="([^"]*)">(.*?)</volume>', txt, re.S):
        vol = html.unescape(vm.group(1))
        for tm in re.finditer(r'<title startparagraph="(\d+)">(.*?)</title>', vm.group(2), re.S):
            out.append({'sp': int(tm.group(1)), 'title': html.unescape(tm.group(2).strip()), 'vol': vol})
    if not out:  # volume 래핑이 없을 때
        for tm in re.finditer(r'<title startparagraph="(\d+)">(.*?)</title>', txt, re.S):
            out.append({'sp': int(tm.group(1)), 'title': html.unescape(tm.group(2).strip()), 'vol': ''})
    return out


def parse_corr(corr_path: Path) -> dict:
    """##N -> {'lines': [(원문, 번역), ...], 'tail': 마지막 원문구}"""
    txt = corr_path.read_text(encoding='utf-8-sig')
    marks = [(int(m.group(1)), m.end(), m.start())
             for m in re.finditer(r'^##(\d+)\s*$', txt, re.M)]
    res = {}
    for i, (n, end, _) in enumerate(marks):
        stop = marks[i + 1][2] if i + 1 < len(marks) else len(txt)
        lines = []
        for ln in txt[end:stop].splitlines():
            t = ln.strip()
            if not t or t == '#':
                continue
            parts = t.split('\t')
            src = parts[0].strip()
            tgt = parts[1].strip() if len(parts) > 1 else ''
            lines.append((src, tgt))
        tail = lines[-1][0] if lines else ''
        res[n] = {'lines': lines, 'tail': tail}
    maxn = marks[-1][0] if marks else 0
    return {'paras': res, 'max': maxn}


def classify(title: str, corr_han: str) -> Tuple[str, str]:
    th = han(strip_num(title))
    ch = corr_han
    if th and ch and (ch == th or ch.startswith(th) or (len(ch) >= 4 and th.startswith(ch))):
        return ('제목', 'tag-title')
    ym = _YEAR.match(th)
    if ym and ch.replace('有', '', 1).startswith(ym.group(1)):
        return ('연도', 'tag-year')
    return ('본문', 'tag-body')


# ---- 대상 탐색 ----

def discover(pc_root: Path) -> List[dict]:
    items = []
    seen = set()

    def add(toc: Path, corr: Optional[Path], book: str, layout: str):
        if book in seen:
            return
        seen.add(book)
        items.append({'book': book, 'toc': toc, 'corr': corr, 'layout': layout})

    # 레이아웃 1: {pc}/{book}/{book}_목차.xml
    for toc in sorted(pc_root.glob('*/*_목차.xml')):
        book = toc.name[:-len('_목차.xml')]
        if toc.parent.name != book:
            continue
        cands = list(toc.parent.glob('*교정*보강문서*.txt'))
        add(toc, cands[0] if cands else None, book, '본편')
    # 레이아웃 2: {pc}/병렬말뭉치 결과물/{book}_결과물/{book}_목차.xml
    rb = pc_root / '병렬말뭉치 결과물'
    if rb.is_dir():
        for toc in sorted(rb.glob('*_결과물/*_목차.xml')):
            book = toc.name[:-len('_목차.xml')]
            cands = list(toc.parent.glob('*교정*보강문서*.txt'))
            add(toc, cands[0] if cands else None, book, '결과물')
    return items


# ---- HTML ----

CSS = """
:root{
  --sans:'Pretendard GOV Variable','Spoqa Han Sans Neo',sans-serif;
  --serif:'Noto Serif CJK KR','Noto Serif KR',serif;
  /* 기본 = 라이트 모드 */
  --bg:#f5f6f8;--card:#ffffff;--line:#e3e6ea;--fg:#1c2128;--mut:#59616c;--acc:#1f6feb;
  --th:#eef1f5;--vol:#eef5ff;--prev:#98a1b0;--rowhover:rgba(31,111,235,.05);
  --bad:#cf222e;--ok:#1a7f37;--shadow:0 1px 2px rgba(0,0,0,.06);
}
[data-theme="dark"]{
  --bg:#0f1115;--card:#171a21;--line:#262b36;--fg:#e6e9ef;--mut:#8b93a7;--acc:#6ea8fe;
  --th:#1c2029;--vol:#11141b;--prev:#5b6270;--rowhover:rgba(110,168,254,.05);
  --bad:#f85149;--ok:#3fb950;--shadow:0 1px 3px rgba(0,0,0,.3);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--sans);font-size:14px;line-height:1.55;-webkit-font-smoothing:antialiased}
a{color:var(--acc);text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:1200px;margin:0 auto;padding:24px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;color:var(--mut);font-weight:500;margin:0 0 18px}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0;color:var(--mut);font-size:13px}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:600;white-space:nowrap}
.tag-title{background:rgba(26,127,55,.12);color:#1a7f37;border:1px solid rgba(26,127,55,.35)}
.tag-year{background:rgba(9,105,218,.12);color:#0969da;border:1px solid rgba(9,105,218,.35)}
.tag-body{background:rgba(107,114,128,.12);color:#57606a;border:1px solid rgba(107,114,128,.35)}
[data-theme="dark"] .tag-title{background:rgba(46,160,67,.18);color:#3fb950;border-color:rgba(46,160,67,.4)}
[data-theme="dark"] .tag-year{background:rgba(59,130,246,.18);color:#6ea8fe;border-color:rgba(59,130,246,.4)}
[data-theme="dark"] .tag-body{background:rgba(139,148,167,.15);color:#a9b1c2;border-color:rgba(139,148,167,.35)}
table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;box-shadow:var(--shadow)}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}
th{position:sticky;top:0;background:var(--th);color:var(--mut);font-weight:600;font-size:12px;z-index:1}
tr:last-child td{border-bottom:none}
tbody tr:hover td{background:var(--rowhover)}
.num{font-variant-numeric:tabular-nums;color:var(--mut);text-align:right;white-space:nowrap}
.sp{font-variant-numeric:tabular-nums;font-weight:700;text-align:right;white-space:nowrap}
.title{font-weight:600}
.han{font-family:var(--serif);font-size:15.5px;line-height:1.5}
.kor{color:var(--mut);font-size:13px;margin-top:2px}
.prev{color:var(--prev);font-size:12px;font-style:italic;border-right:2px solid var(--line);padding-right:8px}
.volrow td{background:var(--vol);color:var(--acc);font-weight:700;font-size:13px}
.summary td,.summary th{padding:7px 10px}
.bad{color:var(--bad);font-weight:700}
.ok{color:var(--ok)}
.pill{font-variant-numeric:tabular-nums}
.foot{color:var(--mut);font-size:12px;margin-top:24px}
.back{display:inline-block;margin-bottom:16px}
.theme-toggle{position:fixed;top:14px;right:16px;z-index:10;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:6px 12px;font-size:13px;font-family:var(--sans);cursor:pointer;box-shadow:var(--shadow)}
.theme-toggle:hover{border-color:var(--acc)}
"""

# 폰트 CDN (로컬 미설치 시 폴백) + 라이트/다크 토글
HEAD_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://spoqa.github.io/spoqa-han-sans/css/SpoqaHanSansNeo.css">'
    '<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard/packages/pretendard-gov/dist/web/variable/pretendardvariable-gov-dynamic-subset.css">'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@400;600&display=swap">'
)
THEME_INIT = ("<script>(function(){var t='light';try{t=localStorage.getItem('tocTheme')||'light'}"
              "catch(e){}document.documentElement.setAttribute('data-theme',t)})();</script>")
THEME_JS = ("<script>function applyTheme(t){document.documentElement.setAttribute('data-theme',t);"
            "try{localStorage.setItem('tocTheme',t)}catch(e){}var b=document.querySelector('.theme-toggle');"
            "if(b)b.textContent=(t==='dark'?'\\u2600\\ufe0f \\ub77c\\uc774\\ud2b8 \\ubaa8\\ub4dc':"
            "'\\ud83c\\udf19 \\ub2e4\\ud06c \\ubaa8\\ub4dc')}"
            "function toggleTheme(){var c=document.documentElement.getAttribute('data-theme')||'light';"
            "applyTheme(c==='dark'?'light':'dark')}"
            "applyTheme(document.documentElement.getAttribute('data-theme')||'light');</script>")


def html_doc(title: str, inner: str) -> str:
    return (
        '<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{esc(title)}</title>' + HEAD_FONTS +
        f'<style>{CSS}</style>' + THEME_INIT +
        '</head><body>'
        '<button class="theme-toggle" onclick="toggleTheme()">테마</button>'
        + inner + THEME_JS +
        '</body></html>')


def esc(s: str) -> str:
    return html.escape(s or '')


def render_book(item: dict) -> Tuple[str, dict]:
    book = item['book']
    toc = parse_toc(item['toc'])
    corr = parse_corr(item['corr']) if item['corr'] else {'paras': {}, 'max': 0}
    paras = corr['paras']
    counts = {'제목': 0, '연도': 0, '본문': 0, '없음': 0}
    rows = []
    last_vol = None
    for i, e in enumerate(toc, 1):
        if e['vol'] != last_vol:
            rows.append(f'<tr class="volrow"><td colspan="6">{esc(e["vol"]) or "（권 미지정）"}</td></tr>')
            last_vol = e['vol']
        p = paras.get(e['sp'])
        if not p or not p['lines']:
            counts['없음'] += 1
            cls = ('없음', 'tag-body')
            cont = '<span class="bad">해당 ##문단 없음/빈 문단</span>'
            prev = ''
        else:
            ch = han(''.join(s for s, _ in p['lines'][:2]))
            cls = classify(e['title'], ch)
            counts[cls[0]] += 1
            shown = p['lines'][:3]
            cont = ''.join(
                f'<div class="han">{esc(s)}</div>' + (f'<div class="kor">{esc(t)}</div>' if t else '')
                for s, t in shown)
            pv = paras.get(e['sp'] - 1)
            prev = f'<div class="prev">…{esc(pv["tail"][-22:])}</div>' if pv and pv['tail'] else ''
        rows.append(
            f'<tr><td class="num">{i}</td>'
            f'<td class="title">{esc(e["title"])}</td>'
            f'<td class="sp">{e["sp"]}</td>'
            f'<td>{prev}</td>'
            f'<td>{cont}</td>'
            f'<td><span class="tag {cls[1]}">{cls[0]}</span></td></tr>')

    body = (
        f'<div class="wrap"><a class="back" href="index.html">← 목록</a>'
        f'<h1>{esc(book)}</h1>'
        f'<h2>{len(toc)}개 항목 · 교정문서 최대 ##{corr["max"]} · '
        f'<span class="tag tag-title">제목 {counts["제목"]}</span> '
        f'<span class="tag tag-year">연도 {counts["연도"]}</span> '
        f'<span class="tag tag-body">본문 {counts["본문"]}</span>'
        + (f' <span class="tag tag-body bad">없음 {counts["없음"]}</span>' if counts['없음'] else '')
        + '</h2>'
        '<div class="legend">'
        '<span><b>startparagraph</b> = 그 작품이 시작하는 교정문서 ##문단 번호</span>'
        '<span><span class="tag tag-title">제목</span> ##내용이 목차 제목과 일치(제목이 문단으로 존재)</span>'
        '<span><span class="tag tag-year">연도</span> 그 해 經文으로 시작(연도책)</span>'
        '<span><span class="tag tag-body">본문</span> 제목 없이 본문으로 시작(序·墓誌銘 등)</span>'
        '</div>'
        '<table><thead><tr>'
        '<th>#</th><th>목차 제목</th><th>start<br>paragraph</th>'
        '<th>직전 문단 끝</th><th>##startparagraph 교정문서 내용 (원문 / 번역)</th><th>구분</th>'
        '</tr></thead><tbody>'
        + ''.join(rows) +
        '</tbody></table>'
        '<div class="foot">생성: toc_dashboard.py — 교정(정보)보강문서를 ground truth로 표시</div>'
        '</div>')
    page = html_doc(f'{book} · 목차 검증', body)
    stat = {'book': book, 'n': len(toc), 'max': corr['max'], 'counts': counts, 'layout': item['layout']}
    return page, stat


def render_index(stats: List[dict]) -> str:
    total_n = sum(s['n'] for s in stats)
    total_bad = sum(s['counts']['없음'] for s in stats)
    rows = []
    for s in sorted(stats, key=lambda x: (x['layout'], x['book'])):
        c = s['counts']
        badcell = f'<span class="bad">{c["없음"]}</span>' if c['없음'] else '<span class="ok">0</span>'
        rows.append(
            f'<tr><td><a href="{esc(s["book"])}.html">{esc(s["book"])}</a></td>'
            f'<td class="num">{s["layout"]}</td>'
            f'<td class="num pill">{s["n"]}</td>'
            f'<td class="num pill">{s["max"]}</td>'
            f'<td class="num pill">{c["제목"]}</td>'
            f'<td class="num pill">{c["연도"]}</td>'
            f'<td class="num pill">{c["본문"]}</td>'
            f'<td class="num pill">{badcell}</td></tr>')
    body = (
        f'<div class="wrap"><h1>목차 startparagraph 검증 대시보드</h1>'
        f'<h2>{len(stats)}개 서명 · 총 {total_n}개 항목 · 빈/누락 {total_bad}건</h2>'
        '<div class="legend">'
        '<span>각 서명을 클릭하면 항목별 startparagraph 가 가리키는 교정문서 ##내용을 확인할 수 있습니다.</span>'
        '</div>'
        '<table class="summary"><thead><tr>'
        '<th>서명</th><th>구분</th><th>항목수</th><th>최대##</th>'
        '<th><span class="tag tag-title">제목</span></th>'
        '<th><span class="tag tag-year">연도</span></th>'
        '<th><span class="tag tag-body">본문</span></th>'
        '<th>빈/누락</th></tr></thead><tbody>'
        + ''.join(rows) +
        '</tbody></table>'
        '<div class="foot">제목/연도/본문은 모두 “작품 시작”을 가리키는 정상 유형입니다. '
        '“빈/누락”이 0이면 모든 startparagraph 가 실제 문단을 가리킴을 뜻합니다.</div>'
        '</div>')
    return html_doc('목차 startparagraph 검증', body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pc-root', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    pc = Path(args.pc_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = discover(pc)
    stats = []
    for it in items:
        page, stat = render_book(it)
        (out / f'{it["book"]}.html').write_text(page, encoding='utf-8')
        stats.append(stat)
        flag = f'  빈/누락 {stat["counts"]["없음"]}' if stat['counts']['없음'] else ''
        print(f'[OK] {it["book"]}: {stat["n"]}항목{flag}')
    (out / 'index.html').write_text(render_index(stats), encoding='utf-8')
    print(f'\n총 {len(stats)}개 서명 → {out / "index.html"}')


if __name__ == '__main__':
    main()
