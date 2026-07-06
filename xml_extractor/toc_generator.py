#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
toc_generator.py — sources 원문/번역문 XML -> 목차정보 XML 생성 (+교정·검증)

배경
----
각 서명 폴더의 `{책}_목차.xml` 은 다음 형식으로 편(작품)의 시작 문단을 가리킨다.

    <classics name="" id="">
      <volume name="宋大家蘇文忠公文抄 卷1 制策">
        <title startparagraph="26">01. 御試制科策一道</title>
      </volume>
      ...
    </classics>

여기서 `startparagraph` 는 교정정보보강문서(완료) `{책}_교정정보보강문서(완료).txt` 의
`##N` 문단 번호를 가리켜야 한다.

핵심 (왜 기존 목차가 틀렸는가)
------------------------------
`##N` 번호는 `scripts/stopword_pipeline/xml_to_unified.py` 의 단락 재구성/재번호
로직으로 산출된다.
  1. parse_xml_phrase_level : <s>/<w> 단위로 추출, para_id = 가장 가까운 '식별자' 또는 <단락 id>
  2. merge_xml_pair         : (s_id, w_id) 숫자 정렬 + 문장별 para_id 통일(원문 우선)
  3. renumber_paragraphs    : para_id 가 '바뀔 때마다' para_seq += 1  ==>  이것이 ##N

즉 集評(집평)·각주처럼 chi 본문 <s><w> 가 없는 단락은 ## 생성에서 제외된다.
기존 일부 목차(예: 당시삼백수)는 단락을 '문서순으로 단순 카운트'하여 이런 비-본문
단락까지 포함했기 때문에 startparagraph 가 ##N 보다 점점 커지는 누적 드리프트가 생겼다.

이 스크립트는 startparagraph 를 위 '동일한' 로직으로 산출하여 교정문서 ##N 과 일치시킨다.

모드
----
  generate : 원문(+번역문) XML 한 쌍 -> 목차 XML
  batch    : sources 디렉터리 일괄 (원문/번역문 자동 페어링)
  verify   : 계산값 vs (기존 목차 / 교정문서 실제 ##) 대조 리포트 (파일 수정 없음)
  correct  : 기존 목차의 startparagraph 만 재계산하여 덮어쓰기 (제목·volume·순서 보존)

번호 산출 3함수(parse_xml_phrase_level / merge_xml_pair / renumber_paragraphs)는
xml_to_unified.py 와 '동일' 해야 결과가 일치하므로, 의존성(unified_io) 없이 표준
라이브러리만으로 자체 포함했다. (원본: head-repo/CSP/.../xml_to_unified.py)

사용 예
-------
  py toc_generator.py generate --source <원문.xml> --translation <번역문.xml> --output <목차.xml>
  py toc_generator.py batch    --source-dir <sources> --output-dir <out> [--books 당시삼백수1 ...]
  py toc_generator.py verify   --source-dir <sources> --pc-root "<...\\PC2025(xlsx)>" [--books ...]
  py toc_generator.py correct  --source-dir <sources> --pc-root "<...\\PC2025(xlsx)>" [--books ...] [--apply]
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ===========================================================================
# 1) 단락 재번호 로직  (xml_to_unified.py 와 동일 — 자체 포함)
# ===========================================================================

def _local(tag: str) -> str:
    """네임스페이스 제거."""
    return tag.split('}')[-1] if isinstance(tag, str) else tag


def parse_xml_phrase_level(xml_path: Path, content_type: str):
    """반환: [(para_id, s_id, w_id, text, doc_order), ...]  (xml_to_unified 와 동일)
    - 한 행 = <w> 하나
    - para_id : 가장 가까운 상위 노드의 '식별자' 또는 <단락 id>
    - lang 필터 : 원문='chi', 번역문='kor' (미지정은 통과)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []
    counter = [0]

    def should_use_lang(lang_value):
        if not lang_value:
            return True
        return lang_value == ('chi' if content_type == '원문' else 'kor')

    def walk(elem, parent_para_ids=None, current_lang=None):
        if parent_para_ids is None:
            parent_para_ids = []
        if current_lang is None:
            current_lang = root.get('lang', '')

        next_para_ids = parent_para_ids.copy()
        if '식별자' in elem.attrib:
            next_para_ids.append(elem.get('식별자'))
        elif _local(elem.tag) == '단락' and 'id' in elem.attrib:
            next_para_ids.append(elem.get('id'))

        if 'lang' in elem.attrib:
            current_lang = elem.get('lang')

        if _local(elem.tag) == 's':
            if should_use_lang(current_lang):
                s_id = elem.get('id', '')
                pid = next_para_ids[-1] if next_para_ids else ''
                if s_id and pid:
                    for w_elem in elem.findall('.//w'):
                        w_id = w_elem.get('id', '')
                        txt = ''.join(w_elem.itertext()).strip()
                        if w_id:
                            counter[0] += 1
                            data.append((pid, s_id, w_id, txt, counter[0]))
            return  # <s> 안으로는 더 들어가지 않음

        for child in elem:
            walk(child, next_para_ids, current_lang)

    walk(root)
    return data


def _normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    t = text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    return ' '.join(t.split())


def _normalize_para_id(para_id: str) -> str:
    if para_id and para_id.endswith('_T'):
        return para_id[:-2]
    return para_id


def merge_xml_pair(source_data, translation_data):
    """(s_id, w_id) 기준 머지 + 문장별 para_id 통일(원문 우선) + (s_id,w_id) 숫자 정렬.
    xml_to_unified.merge_xml_pair 와 동일.
    """
    merged = {}

    for pid, s_id, w_id, txt, _ in source_data:
        pid = _normalize_para_id(pid)
        k = (s_id, w_id)
        e = merged.setdefault(k, {})
        if not e.get('para_id'):
            e['para_id'] = pid
        e.setdefault('s_id', s_id)
        e.setdefault('w_id', w_id)
        e['원문'] = txt

    for pid, s_id, w_id, txt, _ in translation_data:
        pid = _normalize_para_id(pid)
        k = (s_id, w_id)
        e = merged.setdefault(k, {})
        if not e.get('para_id'):
            e['para_id'] = pid
        e.setdefault('s_id', s_id)
        e.setdefault('w_id', w_id)
        e['번역문'] = txt

    rows = []
    for k, e in merged.items():
        rows.append({
            'para_id': e.get('para_id', ''),
            's_id': e['s_id'],
            'w_id': e['w_id'],
            '원문': _normalize_text(e.get('원문', '')),
            '번역문': _normalize_text(e.get('번역문', '')),
        })

    # 문장(s_id) 단위 para_id 통일 : 원문 측 para_id 우선
    sent_pid = {}
    for r in rows:
        if r['원문']:
            sent_pid.setdefault(r['s_id'], r['para_id'])
    for r in rows:
        if r['s_id'] not in sent_pid:
            sent_pid[r['s_id']] = r['para_id']
    for r in rows:
        r['para_id'] = sent_pid[r['s_id']]

    rows.sort(key=lambda r: (
        int(r['s_id']) if str(r['s_id']).isdigit() else 0,
        int(r['w_id']) if str(r['w_id']).isdigit() else 0,
    ))
    return rows


def renumber_paragraphs(rows):
    """para_id 가 바뀔 때마다 새 번호. 같은 식별자 연속은 같은 번호. (== ##N)"""
    current = 0
    prev = None
    out = []
    for r in rows:
        if r['para_id'] != prev:
            current += 1
            prev = r['para_id']
        nr = dict(r)
        nr['para_seq'] = current
        out.append(nr)
    return out


def build_sid_to_seq(source_xml: Path, translation_xml: Optional[Path]):
    """반환: (s_id -> para_seq 매핑 dict, 최대 para_seq).
    교정문서 ##N 과 동일한 문단 번호 체계.
    번역문이 없으면 원문만으로 계산(가능하면 번역문 포함을 권장).
    """
    src = parse_xml_phrase_level(source_xml, '원문')
    tgt = parse_xml_phrase_level(translation_xml, '번역문') if translation_xml and translation_xml.exists() else []
    rows = renumber_paragraphs(merge_xml_pair(src, tgt))
    sid2seq: Dict[str, int] = {}
    maxseq = 0
    for r in rows:
        sid = r['s_id']
        if sid not in sid2seq:
            sid2seq[sid] = r['para_seq']
        maxseq = max(maxseq, r['para_seq'])
    return sid2seq, maxseq


# ===========================================================================
# 2) 제목(작품) 추출  — 문서 순서로 <제목> 과 <s> 를 훑어 작품별 문장 id 수집
# ===========================================================================

class Work:
    __slots__ = ('level', 'title', 'parent', 'sids')

    def __init__(self, level, title, parent):
        self.level = level
        self.title = title
        self.parent = parent
        self.sids: List[str] = []


def extract_works(source_xml: Path) -> List[Work]:
    """문서 순서대로 <제목> 을 만나면 새 작품 시작, 이어지는 <s> 들의 id 를 그 작품에 귀속.
    - level 스택으로 상위(volume) 제목 추적
    - 내용이 없는 제목(바로 다음에 하위 제목이 오는 경우)은 sids 가 비어 volume 으로 간주
    """
    root = ET.parse(source_xml).getroot()
    works: List[Work] = []
    level_stack: List[Tuple[int, str]] = []
    cur: List[Optional[Work]] = [None]

    def walk(elem):
        tag = _local(elem.tag)
        if tag == '제목':
            try:
                lvl = int(elem.get('level') or 0)
            except ValueError:
                lvl = 0
            txt = ''.join(elem.itertext()).strip()
            txt = re.sub(r'\s+', ' ', txt)
            while level_stack and level_stack[-1][0] >= lvl:
                level_stack.pop()
            parent = level_stack[-1][1] if level_stack else ''
            level_stack.append((lvl, txt))
            w = Work(lvl, txt, parent)
            works.append(w)
            cur[0] = w
            return  # 제목 하위는 보지 않음 (제목 안 텍스트는 위에서 수집 완료)
        if tag == 's':
            sid = elem.get('id')
            if sid and cur[0] is not None:
                cur[0].sids.append(sid)
            return
        for ch in elem:
            walk(ch)

    walk(root)
    return works


# ===========================================================================
# 3) 목차 계산 / 출력
# ===========================================================================

class TocEntry:
    __slots__ = ('volume', 'title', 'startparagraph')

    def __init__(self, volume, title, startparagraph):
        self.volume = volume
        self.title = title
        self.startparagraph = startparagraph


def compute_toc(source_xml: Path, translation_xml: Optional[Path],
                skip_frontmatter: bool = False) -> Tuple[List[TocEntry], int]:
    """반환: (목차 항목 리스트, 최대 ##N).
    skip_frontmatter=True 면 상위 volume 이 없는(parent=='') 최상위 제목(解題·凡例·
    引·本傳·圖·序 등 권두 자료)을 목차에서 제외 — 기존 목차의 큐레이션과 일치시킬 때 사용.
    """
    sid2seq, maxseq = build_sid_to_seq(source_xml, translation_xml)
    works = extract_works(source_xml)
    entries: List[TocEntry] = []
    for w in works:
        seqs = [sid2seq[s] for s in w.sids if s in sid2seq]
        if not seqs:
            continue  # 내용 없는 제목(volume) 또는 비-본문 -> 목차 항목 아님
        if skip_frontmatter and not w.parent:
            continue  # 권두 자료(상위 volume 없음) 제외
        entries.append(TocEntry(w.parent, w.title, min(seqs)))
    return entries, maxseq


def _xml_escape(s: str) -> str:
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
             .replace('"', '&quot;'))


def write_toc_xml(entries: List[TocEntry], out_path: Path,
                  classics_name: str = '', classics_id: str = '') -> None:
    lines = ["<?xml version='1.0' encoding='utf-8'?>",
             f'<classics name="{_xml_escape(classics_name)}" id="{_xml_escape(classics_id)}">']
    for e in entries:
        lines.append(f'  <volume name="{_xml_escape(e.volume)}">')
        lines.append(f'    <title startparagraph="{e.startparagraph}">{_xml_escape(e.title)}</title>')
        lines.append('  </volume>')
    lines.append('</classics>')
    out_path.write_text('\n'.join(lines), encoding='utf-8')


# ===========================================================================
# 4) 교정문서(완료) 기반 실제 ##N 확인 (독립 검증 — ground truth)
# ===========================================================================

_CJK_HAN = re.compile(r'[㐀-䶿一-鿿가-힣]')


def _nf(s: str) -> str:
    """CJK + 한글만 남김 (번호·〈〉·표점·B.C.·공백 제거). 현토(한글)는 식별자로 유지."""
    return ''.join(_CJK_HAN.findall(s or ''))


def parse_corr_paragraphs(corr_path: Path) -> List[Tuple[int, str]]:
    """교정문서(완료) -> [(N, 원문정규화), ...]  (원문 = 탭 왼쪽)
    utf-8-sig 로 읽어 선두 BOM(\\ufeff)을 제거 — BOM 이 있으면 첫 마커 ##1 의
    '^##' 매칭이 깨져 ##1 을 놓치므로 반드시 필요.
    """
    txt = corr_path.read_text(encoding='utf-8-sig')
    marks = [(int(m.group(1)), m.end(), m.start())
             for m in re.finditer(r'^##(\d+)\s*$', txt, re.M)]
    out = []
    for i, (n, end, _) in enumerate(marks):
        stop = marks[i + 1][2] if i + 1 < len(marks) else len(txt)
        buf = []
        for ln in txt[end:stop].splitlines():
            t = ln.strip()
            if not t or t == '#':
                continue
            buf.append(t.split('\t')[0])
        out.append((n, _nf(''.join(buf))[:200]))
    return out


def true_start_in_corr(paras: List[Tuple[int, str]], title: str,
                       body_probe: str, last_n: int) -> Tuple[Optional[int], bool]:
    """교정문서에서 작품 시작 ##N 을 찾는다.
    1) 제목이 헤딩으로 존재하면 그 ##N (title_present=True)
    2) 없으면 본문 첫 원문(body_probe)을 포함하는 ##N (title_present=False)
    last_n 이후만 탐색(순서 보존).
    """
    tn = _nf(title)
    if tn:
        for n, pnf in paras:
            if n <= last_n:
                continue
            if pnf == tn or (len(tn) >= 3 and pnf.startswith(tn)):
                return n, True
    bp = _nf(body_probe)[:16]
    if bp:
        for n, pnf in paras:
            if n <= last_n:
                continue
            if bp in pnf:
                return n, False
    return None, False


def _first_body_probe(source_xml: Path, work_title: str, work_sids: List[str]) -> str:
    """작품의 첫 본문 원문(제목 단락 제외) 일부를 반환 — 교정문서 본문 앵커용.
    여기서는 간단히 work 의 sids 순서로 첫 비-제목 문장을 못 구하므로,
    호출부에서 corr 매칭은 title 우선 + 실패 시 body 로 처리한다.
    (본문 프로브가 필요 없으면 빈 문자열)
    """
    return ''


# ===========================================================================
# 5) 책 이름 / 경로 유틸
# ===========================================================================

def extract_book_name(filename: str) -> Optional[str]:
    """jti_1e0208-[역주]춘추좌씨전8_원문_x-C2017.xml -> 춘추좌씨전8"""
    m = re.search(r'\[(?:역주|현토)\](.+?)_(?:원문|번역문)', filename)
    return m.group(1) if m else None


def find_pairs(source_dir: Path, books: Optional[List[str]] = None) -> List[Tuple[str, Path, Path]]:
    """(book, 원문, 번역문) 페어 목록."""
    pairs = []
    for src in sorted(source_dir.glob('*_원문*.xml')):
        book = extract_book_name(src.name)
        if not book:
            continue
        if books and book not in books:
            continue
        tgt = src.parent / src.name.replace('원문', '번역문')
        pairs.append((book, src, tgt if tgt.exists() else None))
    return pairs


def _result_dir(pc_root: Path, book: str) -> Path:
    return pc_root / '병렬말뭉치 결과물' / f'{book}_결과물'


def find_corr_file(pc_root: Path, book: str) -> Optional[Path]:
    """교정(정보)보강문서 위치 — 두 레이아웃/파일명 모두 지원.
      1) {pc_root}/{book}/{book}_교정정보보강문서(완료).txt
      2) {pc_root}/병렬말뭉치 결과물/{book}_결과물/{book}_교정·정보보강문서.txt
    """
    p = pc_root / book / f'{book}_교정정보보강문서(완료).txt'
    if p.exists():
        return p
    rd = _result_dir(pc_root, book)
    if rd.is_dir():
        cands = sorted(rd.glob('*교정*보강문서*.txt'))  # '교정정보…' / '교정·정보…' 모두 매칭
        if cands:
            return cands[0]
    return None


def find_existing_toc(pc_root: Path, book: str) -> Optional[Path]:
    """목차.xml 위치 — 두 레이아웃 모두 지원."""
    p = pc_root / book / f'{book}_목차.xml'
    if p.exists():
        return p
    p2 = _result_dir(pc_root, book) / f'{book}_목차.xml'
    if p2.exists():
        return p2
    return None


def align_new_sps(existing: List[Tuple[int, str, str]],
                  computed: List[TocEntry]) -> List[dict]:
    """기존 목차 항목 순서대로 [{old, new, title, matched}] 반환.
    - 항목수가 같으면 순서(order) 정렬 (한국어 제목 책 포함 안전)
    - 다르면 제목(CJK+한글 정규화) 전진(cursor) 정렬 — 중복 제목/서문 삽입 대응
    matched=False 면 대응 작품을 못 찾은 것(old 유지).
    """
    out = []
    if len(existing) == len(computed):
        for (old, title, _vol), ce in zip(existing, computed):
            out.append({'old': old, 'new': ce.startparagraph, 'title': title, 'matched': True})
        return out
    # 제목 정렬
    used = [False] * len(computed)
    cursor = 0
    for (old, title, _vol) in existing:
        ex = _nf(title)
        found = None
        for j in range(cursor, len(computed)):
            if used[j]:
                continue
            cj = _nf(computed[j].title)
            if ex and cj and (ex == cj or (min(len(ex), len(cj)) >= 3
                                           and (cj.startswith(ex) or ex.startswith(cj)))):
                found = j
                break
        if found is None:  # cursor 이전 포함 전체 재탐색(순서 약간 어긋난 경우)
            for j in range(len(computed)):
                if used[j]:
                    continue
                cj = _nf(computed[j].title)
                if ex and cj and ex == cj:
                    found = j
                    break
        if found is not None:
            used[found] = True
            cursor = found + 1
            out.append({'old': old, 'new': computed[found].startparagraph,
                        'title': title, 'matched': True})
        else:
            out.append({'old': old, 'new': old, 'title': title, 'matched': False})
    return out


def parse_existing_toc(toc_path: Path) -> List[Tuple[int, str, str]]:
    """기존 목차 -> [(startparagraph, title, volume), ...]"""
    txt = toc_path.read_text(encoding='utf-8')
    out = []
    for vm in re.finditer(r'<volume name="([^"]*)">(.*?)</volume>', txt, re.S):
        vol = vm.group(1)
        for tm in re.finditer(r'<title startparagraph="(\d+)">(.*?)</title>', vm.group(2), re.S):
            out.append((int(tm.group(1)), tm.group(2).strip(), vol))
    # volume 래핑이 없을 수도 있으니 fallback
    if not out:
        for tm in re.finditer(r'<title startparagraph="(\d+)">(.*?)</title>', txt, re.S):
            out.append((int(tm.group(1)), tm.group(2).strip(), ''))
    return out


# ===========================================================================
# 6) 모드 구현
# ===========================================================================

def cmd_generate(args):
    src = Path(args.source)
    trans = Path(args.translation) if args.translation else None
    entries, maxseq = compute_toc(src, trans, skip_frontmatter=args.skip_frontmatter)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_toc_xml(entries, out, args.name or '', args.id or '')
    print(f'[OK] {out.name}: {len(entries)}개 항목, 최대 ##{maxseq} -> {out}')


def cmd_batch(args):
    source_dir = Path(args.source_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = find_pairs(source_dir, args.books)
    print(f'대상 {len(pairs)}개\n')
    for book, src, trans in pairs:
        try:
            entries, maxseq = compute_toc(src, trans, skip_frontmatter=args.skip_frontmatter)
            out = out_dir / f'{book}_목차.xml'
            write_toc_xml(entries, out)
            warn = '' if trans else '  (번역문 없음 — 원문만)'
            print(f'[OK] {book}: {len(entries)}항목, 최대 ##{maxseq}{warn}')
        except Exception as e:
            print(f'[FAIL] {book}: {e}')


def cmd_verify(args):
    source_dir = Path(args.source_dir)
    pc_root = Path(args.pc_root) if args.pc_root else None
    pairs = find_pairs(source_dir, args.books)
    print('book | 계산항목 | 최대## | 기존목차 | 기존과불일치 | 교정문서대조 | 비고')
    grand = 0
    for book, src, trans in pairs:
        try:
            entries, maxseq = compute_toc(src, trans)
        except Exception as e:
            print(f'{book} | 계산실패: {e}')
            continue
        line = [book, str(len(entries)), str(maxseq)]

        # (a) 기존 목차와 대조 (순서 기반)
        existing = None
        if pc_root:
            tp = find_existing_toc(pc_root, book)
            if tp:
                existing = parse_existing_toc(tp)
        detail_lines = []
        if existing is not None:
            aligned = align_new_sps(existing, entries)
            diff = [(i, a) for i, a in enumerate(aligned) if a['matched'] and a['old'] != a['new']]
            unmatched = sum(1 for a in aligned if not a['matched'])
            cnt_note = '' if len(existing) == len(entries) else f' (항목수 {len(existing)}≠{len(entries)})'
            extra = f' +미매칭{unmatched}' if unmatched else ''
            line += [str(len(existing)), str(len(diff)) + cnt_note + extra]
            grand += len(diff)
            if args.show and diff:
                for i, a in diff[:args.show]:
                    detail_lines.append(f'    └ #{i+1} 기존 sp={a["old"]} → 올바른 ##{a["new"]}  {a["title"][:24]}')
        else:
            line += ['-', '-']

        # (b) 교정문서(완료) 실제 ## 와 대조 (제목 앵커, 독립 ground truth)
        corr_note = '-'
        if pc_root:
            cf = find_corr_file(pc_root, book)
            if cf:
                paras = parse_corr_paragraphs(cf)
                last = 0
                mism = 0
                checked = 0
                for e in entries:
                    tn, present = true_start_in_corr(paras, e.title, '', last)
                    if tn is not None:
                        checked += 1
                        if tn != e.startparagraph:
                            mism += 1
                        last = tn
                corr_note = f'{checked}편확인/{mism}불일치'
        line.append(corr_note)
        print(' | '.join(line))
        for d in detail_lines:
            print(d)
    if grand:
        print(f'\n총 기존-계산 불일치: {grand}건')


def cmd_correct(args):
    source_dir = Path(args.source_dir)
    pc_root = Path(args.pc_root)
    pairs = find_pairs(source_dir, args.books)
    for book, src, trans in pairs:
        tp = find_existing_toc(pc_root, book)
        if not tp:
            print(f'[SKIP] {book}: 기존 목차 없음')
            continue
        try:
            entries, _ = compute_toc(src, trans)
        except Exception as e:
            print(f'[FAIL] {book}: 계산 실패 {e}')
            continue
        existing = parse_existing_toc(tp)
        aligned = align_new_sps(existing, entries)
        unmatched = sum(1 for a in aligned if not a['matched'])
        # 제목/volume 보존, startparagraph 만 교체 (등장 순서대로)
        txt = tp.read_text(encoding='utf-8')
        idx = [0]
        changed = [0]

        def repl(m):
            a = aligned[idx[0]]
            idx[0] += 1
            new = a['new']
            if int(m.group(1)) != new:
                changed[0] += 1
            return f'<title startparagraph="{new}">'

        new_txt = re.sub(r'<title startparagraph="(\d+)">', repl, txt)
        warn = f' (미매칭 {unmatched}개는 유지)' if unmatched else ''
        cnt = '' if len(existing) == len(entries) else f' [항목수 {len(existing)}≠{len(entries)}]'
        if args.apply:
            if changed[0]:
                bak = tp.with_suffix('.xml.bak')
                if not bak.exists():
                    bak.write_text(txt, encoding='utf-8')  # 최초 1회 원본 백업
                tp.write_text(new_txt, encoding='utf-8')
                print(f'[FIX] {book}: {changed[0]}개 startparagraph 수정 적용 (백업: {bak.name}){warn}{cnt}')
            else:
                print(f'[OK ] {book}: 수정 불필요 (이미 정확){warn}{cnt}')
        else:
            print(f'[DRY] {book}: {changed[0]}개 startparagraph 수정 예정 (--apply 로 반영){warn}{cnt}')


# ===========================================================================
# 7) CLI
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description='sources 원문/번역문 XML -> 목차정보 XML (생성·검증·교정)',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    g = sub.add_parser('generate', help='단일 페어 -> 목차 XML')
    g.add_argument('--source', required=True)
    g.add_argument('--translation')
    g.add_argument('--output', required=True)
    g.add_argument('--name', default='')
    g.add_argument('--id', default='')
    g.add_argument('--skip-frontmatter', action='store_true',
                   help='권두 자료(解題·凡例·引·本傳·圖·序 등 상위 volume 없는 제목) 제외')
    g.set_defaults(func=cmd_generate)

    b = sub.add_parser('batch', help='sources 디렉터리 일괄')
    b.add_argument('--source-dir', required=True)
    b.add_argument('--output-dir', required=True)
    b.add_argument('--books', nargs='+')
    b.add_argument('--skip-frontmatter', action='store_true',
                   help='권두 자료 제외')
    b.set_defaults(func=cmd_batch)

    v = sub.add_parser('verify', help='계산값 vs 기존목차/교정문서 대조')
    v.add_argument('--source-dir', required=True)
    v.add_argument('--pc-root', help='PC2025(xlsx) 루트 (기존 목차/교정문서 비교용)')
    v.add_argument('--books', nargs='+')
    v.add_argument('--show', type=int, default=0, help='책별 불일치 상세 N건 출력')
    v.set_defaults(func=cmd_verify)

    c = sub.add_parser('correct', help='기존 목차 startparagraph 만 재계산하여 수정')
    c.add_argument('--source-dir', required=True)
    c.add_argument('--pc-root', required=True)
    c.add_argument('--books', nargs='+')
    c.add_argument('--apply', action='store_true', help='실제 파일에 반영(미지정 시 dry-run)')
    c.set_defaults(func=cmd_correct)

    # Windows 콘솔(cp949)에서도 CJK/한글 출력이 깨지지 않도록 UTF-8 강제
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
