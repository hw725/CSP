#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
인코딩 복구 스크립트 - CP949/UTF-8/EUC-KR 조합
"""
import os
import re
import glob

def try_all_encodings(raw_bytes):
    """모든 가능한 인코딩 조합 시도"""
    results = []
    
    encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-16-le', 'utf-16-be', 'latin-1', 'iso-8859-1']
    
    for enc1 in encodings:
        try:
            step1 = raw_bytes.decode(enc1, errors='ignore')
            for enc2 in encodings:
                try:
                    step2 = step1.encode(enc2, errors='ignore')
                    for enc3 in encodings:
                        try:
                            step3 = step2.decode(enc3, errors='ignore')
                            # 한글 비율 계산
                            korean_chars = len(re.findall(r'[가-힣]', step3))
                            total_chars = len(step3)
                            if total_chars > 0:
                                ratio = korean_chars / total_chars
                                if ratio > 0.1:  # 10% 이상 한글
                                    results.append((ratio, f'{enc1}->{enc2}->{enc3}', step3[:500]))
                        except:
                            pass
                except:
                    pass
        except:
            pass
    
    # 한글 비율 순으로 정렬
    results.sort(reverse=True)
    return results[:10]

def fix_mojibake_patterns(text):
    """알려진 깨진 패턴 수정"""
    # EUC-KR/CP949 깨짐 패턴 -> 원본 한글 매핑
    # 이 매핑은 손상 패턴 분석을 통해 확장 필요
    
    patterns = {
        # 기본 단어들 (분석된 패턴)
        '전근대': '전근대',
        '이전': '이전', 
        '기준': '기준',
        '현토': '현토',
        '분류': '분류',
        '결과': '결과',
        '이삼환': '이삼환',
        '구두지남': '구두지남',
        '임규직': '임규직',
        '구두해법': '구두해법',
        '박문호': '박문호',
        '이두해': '이두해',
        '존칭': '존칭',
        '변이형': '변이형',
        '축약': '축약',
        '포함': '포함',
        '고유': '고유',
        '마커': '마커',
        '총빈도': '총빈도',
        '빈도': '빈도',
        '확장판': '확장판',
        
        # 분류 범주
        '가정': '가정',
        '감탄': '감탄', 
        '개괄': '개괄',
        '객체': '객체',
        '하늘': '하늘',
        '과거': '과거',
        '러니': '러니',
        '나열': '나열',
        '단사': '단사',
        '기사지단': '기사지단',
        '미절': '미절',
        '서술지단': '서술지단',
        '유사이단': '유사이단',
        '쾌절': '쾌절',
        '미래': '미래',
        '리니': '리니',
        '리라': '리라',
        '미분류': '미분류',
        '복합': '복합',
        '상반': '상반',
        '호되': '호되',
        '하단': '하단',
        '으로': '으로',
        '승상': '승상',
        '이니': '이니',
        '하니': '하니',
        '양보': '양보',
        '호만': '호만',
        '이나이라도': '이나이라도',
        '수역': '수역',
        '의문': '의문',
        '설명': '설명',
        '판정': '판정',
        '이유': '이유',
        '하새라': '하새라',
        '일의적승': '일의적승',
        '하야': '하야',
        '인칭': '인칭',
        '하노라': '하노라',
        '인층': '인층',
        '제외': '제외',
        '구두점': '구두점',
        '주체': '주체',
        '하라': '하라',
        '직접인가': '직접인가',
        '직하': '직하',
        '조사': '조사',
        '진행': '진행',
        '유새': '유새',
        '청소': '청소',
        '청원': '청원',
        '하라': '하라',
        '필수조건': '필수조건',
        '이야': '이야',
        
        # 메타 정보
        '출처': '출처',
        '변이형': '변이형',
        '번': '번',
        '개': '개',
        '리오': '리오',
        '니라': '니라',
        '어여': '어여',
        '이여': '이여',
        '인저': '인저',
        '아여': '아여',
        '보적': '보적',
        '무음': '무음',
        '개활': '개활',
        '행위': '행위',
        '주어': '주어',
        '전환': '전환',
        '구두거요': '구두거요',
        '회상': '회상',
        
        # 마커들
        '하면': '하면',
        '어든': '어든',
        '어든': '어든',
        '거든': '거든',
        '건대': '건대',
        '거면': '거면',
        '아든': '아든',
        '옵하면': '옵하면',
        '을면': '을면',
        '건하면': '건하면',
        '거시어': '거시어',
        '고하면': '고하면',
        '이어늘': '이어늘',
        '옵어든': '옵어든',
        '이시어든': '이시어든',
        '옵어늘': '옵어늘',
        '이신대': '이신대',
        '하다': '하다',
        '하시나': '하시나',
        '건하시다': '건하시다',
        '옵하시다': '옵하시다',
        '로다': '로다',
        '놋다': '놋다',
        '도다': '도다',
        '러라': '러라',
        '더라': '더라',
        '면': '면',
        '이시어': '이시어',
        '어대': '어대',
        '전통': '전통',
        '감격': '감격',
        '표현': '표현',
        '특승': '특승',
        '고': '고',
        '니이다': '니이다',
        
        # 추가 패턴
        '상세': '상세',
        '조건': '조건',
        '요약': '요약',
        
        # 스크립트 docstring
        '분류를': '분류를',
        '개선판': '개선판',
        '성을': '성을',
        
        # 단일 문자 수정
        '이고': '이고',
        
        # 추가 분류 용어
        '상반_호되': '상반_호되',
        '하단_으로': '하단_으로',
        '승상_이니': '승상_이니',
        '승상_하니': '승상_하니',
        '양보_호만': '양보_호만',
        '양보_이나이라도': '양보_이나이라도',
        '수역_니': '수역_니',
        '의문_설명': '의문_설명',
        '의문_판정': '의문_판정',
        '이유_하새라': '이유_하새라',
        '일의적승_하야': '일의적승_하야',
        '인칭_하노라': '인칭_하노라',
        '제외_구두점': '제외_구두점',
        '주체_하라': '주체_하라',
        '직하_조사': '직하_조사',
        '진행_유새': '진행_유새',
        '청소_라': '청소_라',
        '청원_하라': '청원_하라',
        '필수조건_이야': '필수조건_이야',
        
        # Python 문법
        '원문': '원문',
        '번역문': '번역문',
        'book_name': 'book_name',
        '전서명': '전서명',
        
        # 더 많은 패턴 추가
        '하고하며': '하고하며',
        '이오이며': '이오이며',
        
        # CLASSIFIED_MARKERS 추가 패턴
        '별': '별',
        '지': '지',
        '가': '가',
        '면': '면',
        '예': '예',
        '청': '청',
        '고': '고',
        '직가': '직가',
        '청소': '청소',
        '청원': '청원',
        '직하': '직하',
        '진행': '진행',
        '《구두지남》': '《구두지남》',
        '임규직《구두해법》': '임규직《구두해법》',
        '변이형': '변이형',
        '별마커': '별마커',
        '별요약': '별요약',
        '기사지단': '기사지단',
        '서술지단': '서술지단',
        '아래': '아래',
        '옵하면': '옵하면',
        '건하면': '건하면',
        '고하면': '고하면',
        '상세': '상세',
        '': '',  # 제어문자 잔여물
        
        # 단일 문자 정리 (마지막에)
        '': '',  # 단독 물음표 제거
    }
    
    # 긴 패턴부터 먼저 치환 (더 정확한 매칭)
    sorted_patterns = sorted(patterns.items(), key=lambda x: len(x[0]), reverse=True)
    
    for broken, fixed in sorted_patterns:
        text = text.replace(broken, fixed)
    
    return text

def recover_file(filepath, output_suffix='_RECOVERED'):
    """파일 복구"""
    print(f'\n=== Processing: {filepath} ===')
    
    with open(filepath, 'rb') as f:
        raw = f.read()
    
    # 제어문자 제거
    if raw and raw[0] == 0x12:
        raw = raw[1:]
        print('Removed leading control char')
    
    # UTF-8로 디코딩
    text = raw.decode('utf-8', errors='replace')
    
    # 패턴 기반 복구
    recovered = fix_mojibake_patterns(text)
    
    # 결과 저장
    base, ext = os.path.splitext(filepath)
    output_path = f'{base}{output_suffix}{ext}'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(recovered)
    
    print(f'Saved: {output_path}')
    try:
        print(f'Preview (first 500 chars):')
        print(recovered[:500].encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
    except:
        print('(Preview unavailable due to encoding)')
    
    return recovered

def main():
    # 모든 MD와 PY 파일 복구
    for pattern in ['*.md', '*.py', 'scripts/*.py']:
        for filepath in glob.glob(pattern):
            if '_RECOVERED' not in filepath and '_FIXED' not in filepath:
                if 'fix_encoding' not in filepath and 'analyze_encoding' not in filepath:
                    recover_file(filepath)

if __name__ == '__main__':
    main()
