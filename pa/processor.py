"""PA 메인 프로세서 - import 문제 해결"""

import sys
import os
from pathlib import Path
import pandas as pd
from typing import List, Dict

# 통합 진행률 관리자
from common.progress_manager import start_unified_progress, update_unified_progress, finish_unified_progress, set_progress_description
# 전역 무결성 검증 모듈
from common.integrity_verifier import verify_global_integrity

# 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

# 로컬 모듈 import
from sentence_splitter import split_target_sentences_advanced
from aligner import compute_similarity_simple
def _normalize_brackets_in_text(text: str) -> str:
    """텍스트의 연쇄/중첩 [-…] 괄호를 정상화한다.
    
    SA의 mask/restore 방식 대신, 정규식으로 직접 제거하되,
    재귀적으로 반복해서 모든 중첩/연쇄 제거.
    
    - 가장 안쪽 중첩부터 제거: [-..[-..]..]
    - 빈 블록 제거: [-], [- ]
    - 공백 정리
    """
    import re
    
    if not text:
        return text
    
    max_iterations = 20
    prev = None
    iteration = 0
    
    while iteration < max_iterations and prev != text:
        prev = text
        iteration += 1
        
        # 1단계: 가장 안쪽 중첩부터 제거
        # 패턴: [-내부[-안쪽]뒷부분]
        # 안쪽 [-...] 블록만 제거하되, 외쪽 괄호 유지
        text = re.sub(r'\[-([^\[\]]*)\[-([^\]]*)\]([^\]]*)\]', r'[-\1\3]', text)
        
        # 2단계: 완전히 비워진 또는 공백만인 블록 제거
        # [-], [- ], [-  ], [- abc - ] 같은 형태들 (공백만 포함)
        text = re.sub(r'\[\-\s*\]', '', text)
        
        # 3단계: 연속된 공백 정리 (마스킹 후 복원 시 발생할 수 있음)
        text = re.sub(r' +', ' ', text)

        # 4단계: 괄호 개수 불균형 보정
        open_cnt = text.count("[-")
        close_cnt = text.count("]")
        # 열림이 더 많으면 앞에서부터 '[-'를 제거
        while open_cnt > close_cnt:
            text = text.replace("[-", "", 1)
            open_cnt -= 1
        # 닫힘이 더 많으면 앞에서부터 ']'를 제거
        while close_cnt > open_cnt:
            text = text.replace("]", "", 1)
            close_cnt -= 1
    
    # 최종 정리
    text = text.strip()
    return text


def _final_cleanup_brackets(text: str) -> str:
    """최종 출력 직전 괄호 블록 중복/불균형을 정리한다.

    - 괄호 내부의 앞뒤 공백 제거 (내용은 유지)
    - 연속된 동일 [-…] 블록이 반복되면 하나만 유지
    - 남는 열림/닫힘 괄호가 있으면 앞에서부터 제거하여 개수 맞춤
    """
    if not text:
        return text

    import re
    
    # 1단계: 괄호 내부의 앞뒤 공백만 제거 (빈 블록은 유지)
    # [-  내용  ] → [-내용]
    def trim_inside(m):
        content = m.group(1).strip()
        return f"[-{content}]"
    text = re.sub(r'\[-([^\]]*)\]', trim_inside, text)
    
    # 2단계: 연속 중복 블록 제거
    pattern = re.compile(r"\[-[^\]]*\]")
    parts = []
    last = 0
    prev_block = None
    for m in pattern.finditer(text):
        # 중간 일반 텍스트
        parts.append(text[last:m.start()])
        block = m.group(0)
        if block == prev_block:
            # 중복 블록은 건너뜀
            last = m.end()
            continue
        parts.append(block)
        prev_block = block
        last = m.end()
    parts.append(text[last:])
    cleaned = ''.join(parts)

    # 3단계: 괄호 개수 불균형 보정
    open_cnt = cleaned.count("[-")
    close_cnt = cleaned.count("]")
    while open_cnt > close_cnt:
        cleaned = cleaned.replace("[-", "", 1)
        open_cnt -= 1
    while close_cnt > open_cnt:
        cleaned = cleaned.replace("]", "", 1)
        close_cnt -= 1

    # 4단계: 연속 공백 정리
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def _ensure_atomic_brackets_in_alignments(alignments: List[Dict]) -> List[Dict]:
    """문장 경계에 걸친 [-…] 괄호 블록을 한 조각에 원자적으로 붙이도록 보정한다.

    - '원문'과 '번역문' 모두에 동일 규칙 적용
    - 결합 텍스트에서 [-…] 블록의 전역 오프셋을 찾고,
      시작/끝이 서로 다른 조각에 걸치면 시작 조각의 끝으로 이동한다.
    - 내부 공백과 원래 표기(괄호 포함)를 그대로 보존한다.
    - 다음 조각이 ']'로 시작하면 닫힘 대괄호를 이전 조각 끝으로 흡수한다.
    - 먼저 텍스트 정규화로 연쇄/중첩 제거.
    """
    import re
    if not alignments:
        return alignments

    def _process_column(col_name: str):
        # 1단계: 각 셀의 텍스트 정규화 (연쇄/중첩 제거)
        for a in alignments:
            if col_name in a and a[col_name]:
                a[col_name] = _normalize_brackets_in_text(str(a[col_name]))
        
        segments = [str(a.get(col_name, '')) for a in alignments]
        if not segments:
            return

        cumulative = [0]
        for s in segments:
            cumulative.append(cumulative[-1] + len(s))

        full = ''.join(segments)
        pattern = re.compile(r"\[-(?:\([^)]*\)|[^\]]*)\]", re.S)
        matches = list(pattern.finditer(full))
        if matches:
            for m in matches:
                start, end = m.start(), m.end()

                def find_idx(pos: int) -> int:
                    for i in range(len(segments)):
                        if cumulative[i] <= pos < cumulative[i+1]:
                            return i
                    return len(segments) - 1

                i_start = find_idx(start)
                i_end = find_idx(end - 1)
                if i_start == i_end:
                    continue

                block = full[start:end]
                local_start = start - cumulative[i_start]
                local_end_in_end_seg = end - cumulative[i_end]

                seg_start = segments[i_start]
                for k in range(i_start + 1, i_end):
                    segments[k] = ''

                seg_end = segments[i_end]
                if local_end_in_end_seg > 0:
                    segments[i_end] = seg_end[local_end_in_end_seg:]

                segments[i_start] = seg_start + block

                cumulative = [0]
                full = ''.join(segments)
                for s in segments:
                    cumulative.append(cumulative[-1] + len(s))

        # 2차 보정: 다음 조각이 ']'로 시작하면 이전 조각으로 흡수
        for i in range(len(segments) - 1):
            nxt = segments[i+1]
            if not nxt:
                continue
            j = 0
            while j < len(nxt) and nxt[j].isspace():
                j += 1
            if j < len(nxt) and nxt[j] == ']':
                rest = nxt[j+1:]
                segments[i] = segments[i] + ']'
                segments[i+1] = rest.lstrip()

        for idx, s in enumerate(segments):
            alignments[idx][col_name] = s

    # 두 컬럼 모두 처리
    _process_column('원문')
    _process_column('번역문')
    return alignments

# === 인용 표지 병합 유틸 ===
def _is_quotation_marker_sentence(text: str) -> bool:
    """번역문 한 줄이 인용 표지(예: '고 하였다', '라고 말한다', '하고 명하셨다', '”고 하였다')만으로 이루어졌는지 판별
    - 닫는 따옴표(", ”, ’) 전후 허용
    - 종결부호(. ? !) 허용
    - 동사/존칭/시제/종결어미 조합 반복(연쇄 마커) 허용
    """
    import re
    if not text or not text.strip():
        return False
    closing_quote = r'["”’]?'
    quotation_particles = r'(고|[이]?라?고|하고|며|면서)'
    speech_verbs = r'(하|말하|말씀하|명하|이르|대답하|답하|묻|문|여쭙|아뢰|전하|칭하|부르|외치)'
    honorific_tense = r'(?:셨|ㅆ|시었|시어|시는|시ㄴ|시ㄹ|시|었|았|였|는|ㄴ|ㄹ|을)?'
    endings = r'(다|ㄴ다|는다|습니다|ㅂ니다|까|ㄹ까|을까|느냐|ㄴ가|는가|라|거라|소|오|어라|아라|니|으니)'
    punctuation = r'[\.。?!,，]?'
    marker_chunk = (
        closing_quote + r'\s*' + quotation_particles + r'\s+' + speech_verbs + honorific_tense + endings + r'\s*' + punctuation + r'\s*' + closing_quote + r'\s*'
    )
    pattern = r'^\s*(?:' + marker_chunk + r')+$'
    return re.match(pattern, text.strip()) is not None


def _merge_quote_marker_rows(alignments: List[Dict]) -> List[Dict]:
    """인용 표지 단독 번역문 행을 직전 행과 병합한다.
    - 번역문/원문 모두 병합하여 전체 텍스트 무결성 유지
    - similarity는 병합 후 간단한 길이 기반으로 재계산
    - 중첩 마커(연속 여러 줄)도 반복 병합
    """
    if not alignments:
        return alignments
    merged: List[Dict] = []
    i = 0
    while i < len(alignments):
        cur = alignments[i]
        # 다음 줄들 중 인용 표지 행을 모두 누적 병합
        j = i + 1
        acc_tgt = []
        acc_src = []
        while j < len(alignments) and _is_quotation_marker_sentence(alignments[j].get('번역문', '')):
            acc_tgt.append(alignments[j].get('번역문', ''))
            # 원문도 함께 병합하여 무결성 유지
            if alignments[j].get('원문', '').strip():
                acc_src.append(alignments[j].get('원문', ''))
            j += 1
        if acc_tgt:
            # 직전 행과 병합
            new_entry = dict(cur)  # shallow copy
            new_entry['번역문'] = (cur.get('번역문', '') + ' ' + ' '.join(acc_tgt)).strip()
            if acc_src:
                new_entry['원문'] = (cur.get('원문', '') + ' ' + ' '.join(acc_src)).strip()
            # 유사도 재계산
            try:
                new_entry['similarity'] = compute_similarity_simple(new_entry.get('원문', ''), new_entry.get('번역문', ''))
            except Exception:
                pass
            merged.append(new_entry)
            i = j  # 병합된 만큼 건너뛰기
        else:
            merged.append(cur)
            i += 1
    return merged

try:
    from aligner import (
        get_embedder_function,
        process_paragraph_alignment,
        restore_paragraph_integrity,
    )
except ImportError as e:
    print(f"❌ aligner import 실패: {e}")
    
    def get_embedder_function(*args, **kwargs):
        print("❌ 임베더 기능을 사용할 수 없습니다.")
        return None

    def process_paragraph_alignment(*args, **kwargs):
        print("❌ 문단 정렬 기능을 사용할 수 없습니다.")
        return []

def process_paragraph_file(
    input_file, 
    output_file, 
    embedder_name="bge", 
    max_length=150, 
    similarity_threshold=0.7,
    openai_model=None,
    openai_api_key=None,
    max_workers=4,      # 🚀 병렬 워커 수 추가
    batch_size=50,      # 🚀 배치 크기 추가
    verbose=False,
    device="cpu"
):
    """입력 엑셀 파일을 읽어 문단 단위로 정렬하고, 결과를 출력 파일로 저장"""
    print(f"📂 PA 파일 처리 시작: {input_file}")
    
    try:
        df = pd.read_excel(input_file)
        print(f"📄 {len(df)}개 문단 로드됨")
    except FileNotFoundError:
        print(f"❌ 입력 파일을 찾을 수 없습니다: {input_file}")
        return None
    except Exception as e:
        print(f"❌ 파일 로드 오류: {e}")
        return None
    
    # 필수 컬럼 확인
    required_columns = ['원문', '번역문']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ 입력 파일에 필수 컬럼이 없습니다: {missing_columns}")
        print(f"📋 현재 컬럼: {list(df.columns)}")
        return None

    # 진행률 초기화
    try:
        start_unified_progress(
            total=len(df),
            description="📊 PA 분할",
            unit="문단",
            bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            mininterval=0.5,
            maxinterval=2.0,
        )
        use_progress_bar = True
    except Exception as e:
        print(f"⚠️ 진행률 초기화 실패: {e}")
        use_progress_bar = False

    all_results: List[Dict] = []
    global_sent_idx = 1  # 전체 문장 번호 연속 부여

    for idx, row in df.iterrows():
        src_paragraph = str(row.get('원문', ''))
        tgt_paragraph = str(row.get('번역문', ''))
        
        if src_paragraph.strip() and tgt_paragraph.strip():
            # 🆕 완전한 문단 정렬 파이프라인 사용 (어절 매칭 포함)
            alignments = process_paragraph_alignment(
                src_paragraph,
                tgt_paragraph,
                embedder_name=embedder_name,
                tokenizer_name='korean_hybrid',
                max_length=max_length,
                similarity_threshold=similarity_threshold,
                device=device,
                quality_threshold=0.8,
                use_spacy_tokenizer=False,
                max_workers=max_workers,
                batch_size=batch_size,
            )

            # 🔧 최종 보정: [-…] 블록을 문장 경계에 걸치지 않도록 원문 조각에 원자적으로 붙임
            try:
                alignments = _ensure_atomic_brackets_in_alignments(alignments)
            except Exception as e:
                if verbose:
                    print(f"⚠️ 괄호 블록 원자화 보정 실패: {e}")
            
            # 🆕 한글 토씨 힌트로 매칭 보정 (기존 로직은 보존)
            try:
                from common.korean_particle_matcher import enhance_pa_alignments_with_particles
                alignments = enhance_pa_alignments_with_particles(alignments)
            except Exception as e:
                if verbose:
                    print(f"⚠️ 토씨 매칭 보정 실패 (기존 결과 유지): {e}")
                # 실패해도 기존 alignments 그대로 사용

            # 🛡️ 최종 무결성 복원: 원문/번역문 결합 텍스트가 입력과 동일하도록 조정
            try:
                alignments = restore_paragraph_integrity(src_paragraph, tgt_paragraph, alignments)
            except Exception as e:
                if verbose:
                    print(f"⚠️ 문단 무결성 복원 실패 (기존 결과 유지): {e}")

            # 🔧 최종 괄호 중복/불균형 정리 (원문/번역문 모두)
            for a in alignments:
                a['원문'] = _final_cleanup_brackets(a.get('원문', ''))
                a['번역문'] = _final_cleanup_brackets(a.get('번역문', ''))
            
            # 🆕 인용 표지 단독 문장 병합 (직전 문장과 결합, 원문/번역문 동시 병합)
            try:
                alignments = _merge_quote_marker_rows(alignments)
            except Exception as e:
                if verbose:
                    print(f"⚠️ 인용 표지 병합 실패 (기존 결과 유지): {e}")
            
            # 문단식별자 추가 + 문장식별자 추가
            original_para_id = row.get('문단식별자', idx + 1)  # 입력의 원본 문단식별자 사용
            for a in alignments:
                a['문단식별자'] = original_para_id
                a['문장식별자'] = global_sent_idx
                global_sent_idx += 1
            
            all_results.extend(alignments)
            
            # 🔧 SA와 동일한 진행률 업데이트
            if use_progress_bar:
                try:
                    update_unified_progress(1, 처리됨=len(all_results))
                except:
                    pass
        
        elif verbose:
            print(f"⚠️ 문단 {idx + 1}: 빈 원문 또는 번역문 건너뜀")
            # 빈 문단도 진행률 업데이트
            if use_progress_bar:
                try:
                    update_unified_progress(1)
                except:
                    pass

        else:
            # 빈 문단도 진행률 업데이트 (비-verbose)
            if use_progress_bar:
                try:
                    update_unified_progress(1)
                except:
                    pass
    
    if not all_results:
        if use_progress_bar:
            try:
                finish_unified_progress("PA 완료 (결과 없음)")
            except:
                pass
        print("❌ 처리된 결과가 없습니다.")
        return None
    
    # 결과 DataFrame 생성
    result_df = pd.DataFrame(all_results)
    
    # 🔧 무결성 확인 후 최종 strip 적용
    if len(result_df) > 0:
        # 원문과 번역문에 대해 strip 적용 (공백 정리)
        if '원문' in result_df.columns:
            result_df['원문'] = result_df['원문'].astype(str).str.strip()
        if '번역문' in result_df.columns:
            result_df['번역문'] = result_df['번역문'].astype(str).str.strip()
        
        if verbose:
            print("✅ 무결성 확인 후 최종 공백 정리 완료")
    
    # 🔧 SA와 동일한 진행률 완료
    if use_progress_bar:
        try:
            finish_unified_progress(f"PA 완료: {len(all_results):,}개 문장 쌍 생성")
        except:
            pass
    
    # 컬럼 순서 정리 - 요구 형식: 문단식별자, 문장식별자, 원문, 번역문, similarity
    final_columns = ['문단식별자', '문장식별자', '원문', '번역문', 'similarity']
    available_columns = [col for col in final_columns if col in result_df.columns]
    result_df = result_df[available_columns]
    
    # 결과 저장
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False, sheet_name='results')

        if verbose:
            print(f"💾 결과 저장: {output_file}")
            print(f"📊 총 {len(all_results)}개 문장 쌍 생성")
            analyze_alignment_results(result_df)
        
        # 🆕 전역 무결성 검증 (정규화 없음, 순수 텍스트 비교)
        try:
            input_df = pd.read_excel(input_file)
            passed, integrity_losses_df, analysis = verify_global_integrity(
                input_df, result_df, 
                source_col='원문', target_col='번역문',
                verbose=verbose
            )
            
            # 무결성 손실 시트를 결과 파일에 추가
            if len(integrity_losses_df) > 0:
                with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
                    integrity_losses_df.to_excel(writer, index=False, sheet_name='integrity_losses')
        except Exception as e:
            if verbose:
                print(f"⚠️ 무결성 검증 오류: {e}")
        
        # 기본 모드에서는 통합 진행률에서 완료 메시지 처리됨

        return result_df

    except Exception as e:
        print(f"❌ 결과 저장 실패: {e}")
        return None

def analyze_alignment_results(result_df: pd.DataFrame):
    """정렬 결과 분석"""
    print("\n📊 정렬 결과 분석:")
    
    # 전체 유사도 분포
    if 'similarity' in result_df.columns:
        print(f"🎯 전체 유사도:")
        print(f"   평균: {result_df['similarity'].mean():.3f}")
        print(f"   최고: {result_df['similarity'].max():.3f}")
        print(f"   최저: {result_df['similarity'].min():.3f}")
        
        # 고품질 매칭 비율
        high_quality = sum(1 for x in result_df['similarity'] if x > 0.7)
        medium_quality = sum(1 for x in result_df['similarity'] if 0.5 <= x <= 0.7)
        low_quality = sum(1 for x in result_df['similarity'] if x < 0.5)
        total = len(result_df)
        
        print(f"📊 품질별 매칭:")
        print(f"   고품질 (>0.7): {high_quality}/{total} ({high_quality/total*100:.1f}%)")
        print(f"   중품질 (0.5-0.7): {medium_quality}/{total} ({medium_quality/total*100:.1f}%)")
        print(f"   저품질 (<0.5): {low_quality}/{total} ({low_quality/total*100:.1f}%)")
    
    # 빈 매칭 확인
    empty_source = sum(1 for x in result_df['원문'] if not str(x).strip())
    empty_target = sum(1 for x in result_df['번역문'] if not str(x).strip())
    
    if empty_source > 0:
        print(f"⚠️ 빈 원문: {empty_source}개")
    if empty_target > 0:
        print(f"⚠️ 빈 번역문: {empty_target}개")
