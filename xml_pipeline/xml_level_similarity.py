#!/usr/bin/env python3
"""
XML 레벨별 유사도 계산기
PA 결과(문장) ↔ XML <s> 단위 비교
SA 결과(구) ↔ XML <w> 단위 비교
"""

import os
import gc
import difflib
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

# XMLUnitParser import
try:
    from .xml_unit_parser import XMLUnitParser
except ImportError:
    from xml_unit_parser import XMLUnitParser

# Sentence Transformers + CUDA 임베딩 모듈 안전 임포트
try:
    import torch
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
    # CUDA 상태 확인
    if torch.cuda.is_available():
        print(f"✅ SentenceTransformers + PyTorch CUDA 지원 (GPU: {torch.cuda.get_device_name()})")
    else:
        print("✅ SentenceTransformers (CPU 모드)")
except ImportError:
    HAS_EMBEDDINGS = False
    torch = None
    print("⚠️ SentenceTransformers 또는 PyTorch를 찾을 수 없습니다. 문자열 유사도만 사용됩니다.")


class XMLLevelSimilarityCalculator:
    """XML 파일과 처리 결과 간 레벨별 유사도 계산 (Integrity 분석 포함)"""
    
    @staticmethod
    def clean_text_for_comparison(text: str) -> str:
        """텍스트 비교용 정제: [ ] - 부호 제거"""
        if not isinstance(text, str):
            text = str(text)
        import re
        # [ ] - 부호 제거
        text = re.sub(r'[\[\-\]]', '', text)
        # 연속된 공백 정리
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def __init__(self, use_embeddings: bool = False):
        self.xml_parser = XMLUnitParser()
        self.use_embeddings = use_embeddings and HAS_EMBEDDINGS
        
        # Integrity 분석용 저장소
        self.integrity_issues = []
        self.integrity_summary = {
            'total_mismatches': 0,
            'severe_issues': 0,
            'moderate_issues': 0,
            'minor_issues': 0,
            'processed_pairs': 0
        }
        
        # 컬럼 표준화를 위한 별칭 집합
        self._id_aliases = {
            'sentence_id': {'sentence_id', '문장식별자', '문장ID', '문장_id', 's_id', 'sent_id'},
            'paragraph_id': {'paragraph_id', '문단식별자', '문단ID', '문단_id', 'p_id', 'para_id'},
            'phrase_id': {'phrase_id', '구식별자', '구ID', '구_id', 'ph_id'},
        }
        self._text_aliases = {
            'original': {'original', '원문', '원문문장', 'source', 'src'},
            'translation': {'translation', '번역', '역문', '번역문', 'trans', 'tgt'},
            'phrase': {'phrase', '구', '구절', '어절', 'segment'},
        }
        
        # GPU 정보 표시
        if self.use_embeddings and torch and torch.cuda.is_available():
            print(f"🚀 PyTorch CUDA GPU 모드 (임베딩 + 매트릭스 가속)")
        else:
            print("🖥️ CPU 전용 모드")
        
        # 임베딩 모델 설정
        self.embedding_model = None
        self.device = 'cuda' if torch and torch.cuda.is_available() and self.use_embeddings else 'cpu'
        self._embedding_cache = {}
        
        if self.use_embeddings:
            try:
                print("🚀 BGE-M3 임베딩 모델 로드 중...")
                self.embedding_model = SentenceTransformer('BAAI/bge-m3', device=self.device)
                print(f"✅ BGE-M3 임베딩 모델 로드 완료 ({self.device})")
                # 임베딩 캐시 디렉토리 설정
                self._cache_dir = "embeddings_cache_openai"
                os.makedirs(self._cache_dir, exist_ok=True)
            except Exception as e:
                print(f"⚠️ BGE-M3 임베딩 모델 로드 실패: {e}")
                print("📝 문자열 기반 유사도로 대체 사용")
                self.use_embeddings = False
                self.embedding_model = None
        else:
            print("📝 문자열 기반 유사도 계산 사용")
    
    def cleanup_resources(self):
        """메모리 및 GPU 리소스 정리"""
        try:
            if hasattr(self, '_embedding_cache'):
                self._embedding_cache.clear()
            
            if hasattr(self, 'embedding_model') and self.embedding_model is not None:
                del self.embedding_model
                self.embedding_model = None
            
            # GPU 메모리 정리
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 가비지 컬렉션
            gc.collect()
            print("🧹 메모리 및 GPU 리소스 정리 완료")
        except Exception as e:
            print(f"⚠️ 리소스 정리 중 오류: {e}")
    
    def calculate_pa_similarity(self, xml_file: str, pa_result_file: str, xml_translation_file: str = None) -> Dict[str, Any]:
        """PA 결과(문장) ↔ XML <s> 단위 간 유사도 계산"""
        print("🔍 PA 레벨 유사도 계산 시작...")
        
        try:
            # 1. XML에서 문장 단위 추출
            xml_sentences = self.xml_parser.extract_sentence_units(xml_file)
            
            # 1-2. XML에서 번역문 단위 추출 (있는 경우)
            xml_translations = None
            if xml_translation_file and Path(xml_translation_file).exists():
                xml_translations = self.xml_parser.extract_sentence_units(xml_translation_file)
                print(f"🔍 번역문 XML 파일 경로 확인: {xml_translation_file}")
            
            # 2. PA 결과 파일 로드
            pa_df = pd.read_excel(pa_result_file)
            
            # 3. PA 결과에서 원문/번역문 컬럼 찾기
            original_col, translation_col = self._find_text_columns(pa_df)
            
            if not original_col or not translation_col:
                return {'error': 'PA 결과 파일에서 원문/번역문 컬럼을 찾을 수 없습니다'}
            
            # 4. PA 유사도 계산
            similarity_results = self._calculate_pa_similarity_scores(
                xml_sentences,
                xml_translations,
                pa_df, 
                original_col, 
                translation_col
            )
            
            print(f"✅ PA 레벨 유사도 계산 완료")
            return similarity_results
            
        except Exception as e:
            error_msg = f"PA 유사도 계산 오류: {e}"
            print(f"❌ {error_msg}")
            return {'error': error_msg}
    
    def calculate_sa_similarity(self, xml_file: str, sa_result_file: str, xml_translation_file: str = None) -> Dict[str, Any]:
        """SA 결과(구) ↔ XML <w> 단위 간 유사도 계산 (<s> 단위별 그룹화)"""
        print("🔍 SA 레벨 유사도 계산 시작...")
        
        try:
            # 1. XML에서 문장별 구 단위 추출 (<s> 단위 고려)
            print(f"📂 XML 파일 경로: {xml_file}")
            print(f"📂 XML 파일 존재 여부: {Path(xml_file).exists()}")
            xml_sentence_groups = self.xml_parser.extract_sentence_grouped_words(xml_file)
            print(f"📊 추출된 XML 문장 그룹 수: {len(xml_sentence_groups) if xml_sentence_groups else 0}")
            
            # 1-2. 번역문 XML 문장별 구 추출 (있는 경우)
            xml_translation_sentence_groups = None
            if xml_translation_file and Path(xml_translation_file).exists():
                xml_translation_sentence_groups = self.xml_parser.extract_sentence_grouped_words(xml_translation_file)
                print(f"🔍 번역문 XML 문장별 구 추출 완료: {len(xml_translation_sentence_groups)}개 문장")
            
            # 2. SA 결과 파일 로드
            sa_df = pd.read_excel(sa_result_file)
            
            # 3. SA 결과에서 원문/번역문/구 컬럼 찾기
            original_col, translation_col = self._find_text_columns(sa_df)
            phrase_col = self._find_phrase_column(sa_df)
            
            if not original_col or not translation_col:
                return {'error': 'SA 결과 파일에서 원문/번역문 컬럼을 찾을 수 없습니다'}
            
            # 4. SA 유사도 계산 (문장 단위별)
            similarity_results = self._calculate_sa_similarity_scores_by_sentence(
                xml_sentence_groups,
                xml_translation_sentence_groups,
                sa_df, 
                original_col, 
                translation_col,
                phrase_col
            )
            
            print(f"✅ SA 레벨 유사도 계산 완료")
            return similarity_results
            
        except Exception as e:
            error_msg = f"SA 유사도 계산 오류: {e}"
            print(f"❌ {error_msg}")
            return {'error': error_msg}
    
    def _find_text_columns(self, df: pd.DataFrame) -> Tuple[str, str]:
        """데이터프레임에서 원문/번역문 컬럼 찾기"""
        original_col = None
        translation_col = None
        
        columns = df.columns.tolist()
        
        # 원문 컬럼 찾기
        for col in columns:
            if '원문' in col or 'original' in col.lower():
                original_col = col
                break
        
        # 번역문 컬럼 찾기
        for col in columns:
            if '번역' in col or '역문' in col or 'translation' in col.lower() or 'trans' in col.lower():
                translation_col = col
                break
        
        return original_col, translation_col
    
    def _find_phrase_column(self, df: pd.DataFrame) -> str:
        """SA 결과에서 구 컬럼 찾기"""
        columns = df.columns.tolist()
        
        for col in columns:
            if '구' in col or 'phrase' in col.lower():
                return col
        
        return None
    
    def _calculate_pa_similarity_scores(self, xml_original_units: List[Dict],
                                      xml_translation_units: List[Dict],
                                      result_df: pd.DataFrame, 
                                      original_col: str, 
                                      translation_col: str) -> Dict[str, Any]:
        """PA 분석: 번역문 기준 매칭"""
        print("🔍 PA 분석 - 번역문 기준 매칭 시작...")
        
        if not xml_translation_units:
            return {'error': '번역문 XML이 제공되지 않았습니다'}
        
        return self._evaluate_with_translation_base(
            xml_original_units, xml_translation_units, result_df, 
            original_col, translation_col, "PA"
        )
    
    def _calculate_sa_similarity_scores(self, xml_original_units: List[Dict],
                                      xml_translation_units: List[Dict],
                                      result_df: pd.DataFrame, 
                                      original_col: str, 
                                      translation_col: str,
                                      phrase_col: str = None) -> Dict[str, Any]:
        """SA 분석: 원문 기준 매칭 (레거시)"""  
        print("🔍 SA 분석 - 원문 기준 매칭 시작...")
        
        return self._evaluate_with_original_base(
            xml_original_units, xml_translation_units, result_df,
            original_col, translation_col, phrase_col, "SA"
        )

    def _calculate_sa_similarity_scores_by_sentence(self, xml_sentence_groups: List[Dict],
                                                  xml_translation_sentence_groups: List[Dict],
                                                  result_df: pd.DataFrame, 
                                                  original_col: str, 
                                                  translation_col: str,
                                                  phrase_col: str = None) -> Dict[str, Any]:
        """SA 분석: 문장 단위별 원문 기준 매칭 (개선된 방식)"""  
        print("🔍 SA 분석 - 문장 단위별 원문 기준 매칭 시작...")
        
        if not xml_sentence_groups:
            return {'error': 'XML 문장 그룹이 없습니다'}
        
        # SA 결과를 문장 단위로 그룹화 (PA 결과 파일에서 문장 정보 추출)
        sa_sentence_groups = self._group_sa_results_by_sentence(result_df, original_col, translation_col)
        
        print(f"📊 XML 문장 그룹: {len(xml_sentence_groups)}개")  
        print(f"📊 SA 문장 그룹: {len(sa_sentence_groups)}개")
        
        # 문장별 매칭 수행
        total_matched_pairs = 0
        total_similarities = []
        all_matched_pairs = []
        
        xml_word_count = sum(len(group['words']) for group in xml_sentence_groups)
        sa_word_count = sum(len(group['phrases']) for group in sa_sentence_groups)
        
        # 번역문 그룹 준비 (있는 경우)
        xml_translation_groups = xml_translation_sentence_groups if xml_translation_sentence_groups else []
        
        # 원문/번역문 개별 유사도 계산을 위한 변수들
        original_similarities = []
        translation_similarities = []
        
        # 각 문장별로 구 매칭 수행
        for xml_group_idx, xml_group in enumerate(xml_sentence_groups):
            xml_sentence_id = xml_group['sentence_id']
            xml_words = xml_group['words']
            
            # 번역문 XML 그룹 (있는 경우)
            xml_translation_group = None
            if xml_group_idx < len(xml_translation_groups):
                xml_translation_group = xml_translation_groups[xml_group_idx]
            
            # 해당하는 SA 문장 그룹 찾기 (문장 순서 기반 매칭)
            matching_sa_group = None
            for sa_group in sa_sentence_groups:
                if self._is_sentence_match(xml_group, sa_group):
                    matching_sa_group = sa_group
                    break
            
            if not matching_sa_group or not matching_sa_group['phrases']:
                # 매칭되는 SA 문장이 없으면 0점 처리
                for xml_word in xml_words:
                    total_similarities.append(0.0)
                    original_similarities.append(0.0)
                    translation_similarities.append(0.0)
                continue
            
            sa_phrases = matching_sa_group['phrases']
            
            # 원문 기준 매칭
            original_alignments = self._find_best_alignment_by_original(
                [word['text'] for word in xml_words],
                [phrase.get('original', '') for phrase in sa_phrases]
            )
            
            # 번역문 기준 매칭 (번역문 XML이 있는 경우)
            translation_alignments = []
            if xml_translation_group and xml_translation_group.get('words'):
                translation_alignments = self._find_best_alignment_by_translation(
                    [word['text'] for word in xml_translation_group['words']],
                    [phrase.get('translation', '') for phrase in sa_phrases]
                )
            
            # 매칭 결과 처리
            for xml_idx, sa_idx, orig_similarity in original_alignments:
                if sa_idx >= 0 and xml_idx < len(xml_words) and sa_idx < len(sa_phrases):
                    total_matched_pairs += 1
                    
                    # SA: 1) 한 세트 유사도, 2) 원문 개별 유사도, 3) 번역문 개별 유사도 모두 계산
                    xml_word = xml_words[xml_idx]
                    sa_phrase = sa_phrases[sa_idx]
                    
                    # 1. 원문 개별 유사도 (기존 계산)
                    original_only_similarity = orig_similarity
                    
                    # 2. 번역문 개별 유사도 계산
                    translation_only_similarity = 0.0
                    if xml_translation_group and xml_idx < len(xml_translation_group.get('words', [])):
                        xml_trans_word = xml_translation_group['words'][xml_idx]
                        sa_trans_text = sa_phrase.get('translation', '')
                        if sa_trans_text:
                            translation_only_similarity = self._calculate_text_similarity(
                                xml_trans_word['text'], sa_trans_text
                            )
                    
                    # 3. 한 세트 유사도 계산 (원문 + 번역문)
                    xml_combined = xml_word['text']  # XML 원문
                    sa_combined = sa_phrase.get('original', '')  # SA 원문
                    
                    # 번역문이 있으면 추가
                    if xml_translation_group and xml_idx < len(xml_translation_group.get('words', [])):
                        xml_trans_word = xml_translation_group['words'][xml_idx]
                        xml_combined += " " + xml_trans_word['text']  # 원문 + 번역문
                    
                    if 'translation' in sa_phrase and sa_phrase['translation']:
                        sa_combined += " " + sa_phrase['translation']  # SA 원문 + 번역문
                    
                    # 한 세트 유사도 계산
                    combined_similarity = self._calculate_text_similarity(xml_combined, sa_combined)
                    
                    # SA는 한 세트 유사도를 메인으로 사용
                    total_similarities.append(combined_similarity)
                    original_similarities.append(original_only_similarity)  # 원문 개별 유사도
                    translation_similarities.append(translation_only_similarity)  # 번역문 개별 유사도
                    
                    all_matched_pairs.append({
                        'xml_idx': f"{xml_sentence_id}_{xml_idx}",
                        'result_idx': f"{matching_sa_group['sentence_id']}_{sa_idx}",
                        'original_similarity': original_only_similarity,  # 원문 개별 유사도
                        'translation_similarity': translation_only_similarity,  # 번역문 개별 유사도
                        'combined_similarity': combined_similarity,  # 한 세트 유사도 (새로 추가)
                        'sentence_id': xml_sentence_id
                    })
                else:
                    total_similarities.append(0.0)
                    original_similarities.append(0.0)
                    translation_similarities.append(0.0)
        
        # 정확도 지표 계산
        precision = total_matched_pairs / sa_word_count if sa_word_count > 0 else 0.0
        recall = total_matched_pairs / xml_word_count if xml_word_count > 0 else 0.0
        f1_score = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
        
        # SA: 한 세트, 원문, 번역문 평균 유사도 계산
        avg_combined_similarity = sum(total_similarities) / len(total_similarities) if total_similarities else 0.0  # 한 세트 유사도
        avg_original_similarity = sum(original_similarities) / len(original_similarities) if original_similarities else 0.0
        avg_translation_similarity = sum(translation_similarities) / len(translation_similarities) if translation_similarities else 0.0
        
        # SA 매칭 과정 integrity 로깅 (문장별 그룹 처리)
        flattened_xml_words = []
        flattened_result_words = []
        for sentence_group in xml_sentence_groups:
            for word in sentence_group.get('words', []):
                flattened_xml_words.append(word)
        
        # SA 결과를 임시로 단어 형태로 변환
        for _, row in result_df.iterrows():
            flattened_result_words.append({
                'original': row.get(original_col, ''),
                'translation': row.get(translation_col, '')
            })
        
        self._log_matching_process(flattened_xml_words, flattened_result_words, 
                                 all_matched_pairs[:50], "SA")  # 상위 50개만 로깅
        
        return {
            'level_type': 'SA',
            'matching_base': 'original', 
            'xml_unit_count': xml_word_count,
            'result_row_count': sa_word_count,
            'matched_pairs': total_matched_pairs,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score, 
            'accuracy': 0.0,  # 정확 매칭은 별도 계산
            'avg_similarity': avg_combined_similarity,  # 한 세트 유사도 (메인)
            'avg_combined_similarity': avg_combined_similarity,  # 한 세트 유사도 (명시)
            'avg_original_similarity': avg_original_similarity,  # 원문 개별 유사도
            'avg_translation_similarity': avg_translation_similarity,  # 번역문 개별 유사도
            'detailed_matches': all_matched_pairs[:100]  # 상위 100개만 저장
        }

    def _evaluate_with_translation_base(self, xml_original_units, xml_translation_units, 
                                       result_df, original_col, translation_col, level_type):
        """번역문 기준 매칭 평가 (PA용)"""
        
        if not xml_translation_units:
            return {'error': '번역문 XML 단위가 없습니다'}
        
        # 컬럼 표준화
        result_df = self._normalize_df_columns(result_df)
        if 'original' not in result_df.columns and original_col in result_df.columns:
            result_df['original'] = result_df[original_col].astype(str)
        if 'translation' not in result_df.columns and translation_col in result_df.columns:
            result_df['translation'] = result_df[translation_col].astype(str)
        
        # XML에서 번역문/원문 추출
        xml_translations = [unit.get('text', '') for unit in xml_translation_units]
        xml_originals = [unit.get('text', '') for unit in xml_original_units]
        
        # 결과 데이터에서 번역문/원문 추출
        result_translations = result_df.get('translation', result_df[translation_col]).fillna('').astype(str).tolist()
        result_originals = result_df.get('original', result_df[original_col]).fillna('').astype(str).tolist()
        
        print(f"📊 번역문 유사도 계산 중... (XML: {len(xml_translations)}, 결과: {len(result_translations)})")
        
        # 번역문 기준 매칭
        alignments = self._find_best_alignment_by_translation(xml_translations, result_translations)
        
        print(f"✅ 번역문 기준 매칭 완료: {len(alignments)}개 매칭 결과")
        
        # 매칭된 쌍에서 한 세트 유사도 평가
        correct_pairs = 0
        similarities = []
        matched_pairs = []
        
        for xml_idx, result_idx, translation_similarity in alignments:
            if result_idx >= 0 and xml_idx < len(xml_originals) and result_idx < len(result_originals):
                xml_orig = xml_originals[xml_idx]
                result_orig = result_originals[result_idx]
                xml_trans = xml_translations[xml_idx] if xml_idx < len(xml_translations) else ''
                result_trans = result_translations[result_idx] if result_idx < len(result_translations) else ''
                
                # PA: 원문+번역문을 한 세트로 유사도 계산
                xml_combined = xml_orig + " " + xml_trans
                result_combined = result_orig + " " + result_trans
                
                combined_similarity = self._calculate_text_similarity(xml_combined, result_combined)
                similarities.append(combined_similarity)
                
                if combined_similarity >= 0.9:
                    correct_pairs += 1
                    
                matched_pairs.append({
                    'xml_idx': xml_idx, 'result_idx': result_idx,
                    'translation_similarity': translation_similarity,  # 번역문 개별 유사도 (참고용)
                    'original_similarity': combined_similarity,  # PA에서는 한 세트 유사도를 저장
                    'combined_similarity': combined_similarity  # 명시적 한 세트 유사도
                })
            else:
                similarities.append(0.0)
        
        # 정확도 지표 계산
        total_xml = len(xml_translations)
        total_result = len(result_translations) 
        matched = len(matched_pairs)
        
        precision = matched / total_result if total_result > 0 else 0.0
        recall = matched / total_xml if total_xml > 0 else 0.0
        f1_score = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
        
        accuracy = correct_pairs / matched if matched > 0 else 0.0
        avg_combined_similarity = sum(similarities) / len(similarities) if similarities else 0.0  # 한 세트 유사도
        
        # 매칭 과정 integrity 로깅
        self._log_matching_process(xml_translation_units, result_df.to_dict('records'), 
                                 matched_pairs, level_type)
        
        return {
            'level_type': level_type,
            'matching_base': 'translation',
            'xml_unit_count': total_xml,
            'result_row_count': total_result,
            'matched_pairs': matched,
            'precision': precision,
            'recall': recall, 
            'f1_score': f1_score,
            'accuracy': accuracy,
            'avg_similarity': avg_combined_similarity,  # PA: 한 세트 유사도 (메인)
            'avg_combined_similarity': avg_combined_similarity,  # 한 세트 유사도 (명시)
            'detailed_matches': matched_pairs
        }
    
    def _evaluate_with_original_base(self, xml_original_units, xml_translation_units, 
                                    result_df, original_col, translation_col, phrase_col, level_type):
        """원문 기준 매칭 평가 (SA용)"""
        
        # 컬럼 표준화
        result_df = self._normalize_df_columns(result_df)
        if 'original' not in result_df.columns and original_col in result_df.columns:
            result_df['original'] = result_df[original_col].astype(str)
        if 'translation' not in result_df.columns and translation_col in result_df.columns:
            result_df['translation'] = result_df[translation_col].astype(str)
        
        # XML에서 원문/번역문 추출
        xml_originals = [unit.get('text', '') for unit in xml_original_units]
        xml_translations = [unit.get('text', '') for unit in xml_translation_units] if xml_translation_units else []
        
        # 결과 데이터에서 원문/번역문 추출
        result_originals = result_df[original_col].fillna('').astype(str).tolist()
        result_translations = result_df[translation_col].fillna('').astype(str).tolist()
        
        print(f"📊 원문 유사도 계산 중... (XML: {len(xml_originals)}, 결과: {len(result_originals)})")
        
        # 원문 기준 매칭
        alignments = self._find_best_alignment_by_original(xml_originals, result_originals)
        
        print(f"✅ 원문 기준 매칭 완료: {len(alignments)}개 매칭 결과")
        
        # 매칭된 쌍에서 번역문 정확도 평가
        correct_pairs = 0
        similarities = []
        matched_pairs = []
        
        for xml_idx, result_idx, original_similarity in alignments:
            if result_idx >= 0 and xml_idx < len(xml_translations) and result_idx < len(result_translations):
                xml_trans = xml_translations[xml_idx] if xml_translations else ''
                result_trans = result_translations[result_idx]
                
                # 번역문 비교 평가
                translation_similarity = self._calculate_text_similarity(xml_trans, result_trans)
                similarities.append(translation_similarity)
                
                if translation_similarity >= 0.9:
                    correct_pairs += 1
                    
                matched_pairs.append({
                    'xml_idx': xml_idx, 'result_idx': result_idx,
                    'original_similarity': original_similarity,
                    'translation_similarity': translation_similarity
                })
            else:
                similarities.append(0.0)
        
        # 정확도 지표 계산
        total_xml = len(xml_originals)
        total_result = len(result_originals)
        matched = len(matched_pairs)
        
        precision = matched / total_result if total_result > 0 else 0.0
        recall = matched / total_xml if total_xml > 0 else 0.0
        f1_score = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
        
        accuracy = correct_pairs / matched if matched > 0 else 0.0
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        return {
            'level_type': level_type,
            'matching_base': 'original', 
            'xml_unit_count': total_xml,
            'result_row_count': total_result,
            'matched_pairs': matched,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score, 
            'accuracy': accuracy,
            'avg_similarity': avg_similarity,
            'detailed_matches': matched_pairs
        }

    def _find_best_alignment_by_translation(self, xml_translations, result_translations):
        """번역문 기준으로 순차적 정렬 찾기"""
        alignments = []
        current_result_idx = 0
        
        for xml_idx in range(len(xml_translations)):
            if current_result_idx >= len(result_translations):
                alignments.append((xml_idx, -1, 0.0))
                continue
                
            xml_text = xml_translations[xml_idx]
            current_result_text = result_translations[current_result_idx]
            
            # 현재 결과와의 유사도 계산
            current_similarity = self._calculate_text_similarity(xml_text, current_result_text)
            
            # 다음 결과와의 유사도 계산 (있는 경우)
            next_similarity = 0.0
            if current_result_idx + 1 < len(result_translations):
                next_result_text = result_translations[current_result_idx + 1]
                next_similarity = self._calculate_text_similarity(xml_text, next_result_text)
            
            # 더 나은 매칭 선택
            if current_similarity >= next_similarity or current_result_idx + 1 >= len(result_translations):
                alignments.append((xml_idx, current_result_idx, current_similarity))
                current_result_idx += 1
            else:
                alignments.append((xml_idx, current_result_idx + 1, next_similarity))
                current_result_idx += 2
        
        return alignments

    def _find_best_alignment_by_original(self, xml_originals, result_originals):
        """원문 기준으로 순차+스마트 매칭 + 부분문자열 감지 스킵 로직"""
        alignments = []
        current_result_idx = 0
        used_xml_texts = []  # 이미 사용된 XML 텍스트들 추적
        
        for xml_idx in range(len(xml_originals)):
            if current_result_idx >= len(result_originals):
                alignments.append((xml_idx, -1, 0.0))
                continue
                
            xml_text = xml_originals[xml_idx]
            
            # 🔧 핵심 로직 1: 현재 SA 구가 이미 앞의 XML에 포함되어 있는지 검사
            current_result_text = result_originals[current_result_idx]
            already_included = False
            
            for prev_xml_text in used_xml_texts:
                if self._is_substring_included(current_result_text, prev_xml_text):
                    # 이미 앞의 XML에 포함되어 있으면 스킵
                    current_result_idx += 1
                    already_included = True
                    break
            
            if already_included:
                # 현재 XML은 매칭 실패로 처리
                alignments.append((xml_idx, -1, 0.0))
                continue
                
            if current_result_idx >= len(result_originals):
                alignments.append((xml_idx, -1, 0.0))
                continue
            
            # 🔧 핵심 로직 2: 동적 스킵 처리 - 다음 SA 구도 검사
            while (current_result_idx < len(result_originals) and 
                   self._should_skip_result(result_originals[current_result_idx], used_xml_texts)):
                current_result_idx += 1
                
            if current_result_idx >= len(result_originals):
                alignments.append((xml_idx, -1, 0.0))
                continue
            
            # 🔧 핵심 로직 3: 스마트한 전방 탐색 (현재 vs 다음 중 선택)
            current_result_text = result_originals[current_result_idx]
            current_similarity = self._calculate_text_similarity(xml_text, current_result_text)
            
            next_similarity = 0.0
            if current_result_idx + 1 < len(result_originals):
                next_result_text = result_originals[current_result_idx + 1]
                # 다음 구도 이미 포함되어 있지 않은지 확인
                if not self._should_skip_result(next_result_text, used_xml_texts):
                    next_similarity = self._calculate_text_similarity(xml_text, next_result_text)
            
            # 더 나은 매칭 선택
            if current_similarity >= next_similarity or current_result_idx + 1 >= len(result_originals):
                alignments.append((xml_idx, current_result_idx, current_similarity))
                used_xml_texts.append(xml_text)  # 사용된 XML 추가
                current_result_idx += 1
            else:
                alignments.append((xml_idx, current_result_idx + 1, next_similarity))
                used_xml_texts.append(xml_text)  # 사용된 XML 추가
                current_result_idx += 2
        
        return alignments
    
    def _is_substring_included(self, sa_text: str, xml_text: str) -> bool:
        """SA 구가 XML 텍스트에 부분문자열로 포함되어 있는지 검사"""
        # 텍스트 정제 후 비교
        clean_sa = self.clean_text_for_comparison(sa_text).replace(' ', '')
        clean_xml = self.clean_text_for_comparison(xml_text).replace(' ', '')
        
        # 너무 짧은 텍스트는 제외 (1-2글자는 우연히 포함될 수 있음)
        if len(clean_sa) <= 2:
            return False
            
        return clean_sa in clean_xml
    
    def _should_skip_result(self, result_text: str, used_xml_texts: List[str]) -> bool:
        """현재 SA 구를 스킵해야 하는지 판단"""
        for xml_text in used_xml_texts:
            if self._is_substring_included(result_text, xml_text):
                return True
        return False

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """텍스트 유사도 계산"""
        if self.use_embeddings and self.embedding_model:
            return self._calculate_embedding_similarity(text1, text2)
        else:
            return self._calculate_string_similarity(text1, text2)

    def _calculate_string_similarity(self, text1: str, text2: str) -> float:
        """문자열 기반 유사도 계산 (텍스트 정제 적용)"""
        clean_text1 = self.clean_text_for_comparison(text1)
        clean_text2 = self.clean_text_for_comparison(text2)
        return difflib.SequenceMatcher(None, clean_text1, clean_text2).ratio()

    def _calculate_embedding_similarity(self, text1: str, text2: str) -> float:
        """임베딩 기반 유사도 계산 (텍스트 정제 적용)"""
        try:
            if not self.embedding_model:
                return self._calculate_string_similarity(text1, text2)
            
            # 텍스트 정제 적용
            clean_text1 = self.clean_text_for_comparison(text1)
            clean_text2 = self.clean_text_for_comparison(text2)
            
            embeddings = self.embedding_model.encode([clean_text1, clean_text2])
            
            if torch:
                emb1 = torch.tensor(embeddings[0]).unsqueeze(0)
                emb2 = torch.tensor(embeddings[1]).unsqueeze(0)
                emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
                emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
                similarity = torch.mm(emb1, emb2.t()).item()
            else:
                # NumPy 계산
                emb1 = embeddings[0] / np.linalg.norm(embeddings[0])
                emb2 = embeddings[1] / np.linalg.norm(embeddings[1])
                similarity = np.dot(emb1, emb2)
            
            return max(0.0, min(1.0, float(similarity)))
            
        except Exception as e:
            print(f"⚠️ 임베딩 유사도 계산 실패, 문자열 유사도로 대체: {e}")
            return self._calculate_string_similarity(text1, text2)

    # ============================================================================
    # Integrity 분석 메서드들
    # ============================================================================
    
    def _analyze_content_integrity(self, xml_text: str, result_text: str, pair_id: str, 
                                 analysis_type: str = "PA") -> Dict[str, Any]:
        """콘텐츠 무결성 분석 (공백 제외 차이점 분석)"""
        # 공백 제외 콘텐츠 비교
        xml_content = ''.join(self.clean_text_for_comparison(xml_text).split())
        result_content = ''.join(self.clean_text_for_comparison(result_text).split())
        
        # 차이점 분석
        diff_ratio = difflib.SequenceMatcher(None, xml_content, result_content).ratio()
        
        # 심각도 분류
        if diff_ratio >= 0.95:
            severity = "minor"
        elif diff_ratio >= 0.80:
            severity = "moderate"  
        else:
            severity = "severe"
        
        # 구체적인 차이점 추출
        differ = difflib.Differ()
        diff_lines = list(differ.compare([xml_content], [result_content]))
        
        missing_chars = []
        extra_chars = []
        
        for line in diff_lines:
            if line.startswith('- '):
                missing_chars.append(line[2:])
            elif line.startswith('+ '):
                extra_chars.append(line[2:])
        
        integrity_data = {
            'pair_id': pair_id,
            'analysis_type': analysis_type,
            'xml_content_length': len(xml_content),
            'result_content_length': len(result_content),
            'content_similarity': diff_ratio,
            'severity': severity,
            'missing_content': missing_chars,
            'extra_content': extra_chars,
            'xml_sample': xml_text[:100] + "..." if len(xml_text) > 100 else xml_text,
            'result_sample': result_text[:100] + "..." if len(result_text) > 100 else result_text
        }
        
        # 전역 카운터 업데이트
        self.integrity_summary['processed_pairs'] += 1
        if diff_ratio < 1.0:
            self.integrity_summary['total_mismatches'] += 1
            self.integrity_summary[f'{severity}_issues'] += 1
            self.integrity_issues.append(integrity_data)
        
        return integrity_data
    

    
        print(f"\n🔍 텍스트 차이 상세 분석:")
        print(f"   원본 길이: {len(xml_text):,}자")
        print(f"   비교 대상 길이: {len(predicted_text):,}자")
        
        # 공백 외 차이점 분석
        xml_clean = ''.join(self.clean_text_for_comparison(xml_text).split())
        predicted_clean = ''.join(self.clean_text_for_comparison(predicted_text).split())
        
        non_space_differences = []
        whitespace_count = 0
        
        if xml_clean == predicted_clean:
            print("✅ 공백 외 차이점 없음 - 차이는 모두 공백/띄어쓰기 관련")
        else:
            print(f"⚠️  공백 외 차이점 발견 ({len(xml_clean) - len(predicted_clean)}개):")
            
            # 구체적인 차이점 찾기
            differ = difflib.Differ()
            diff = list(differ.compare(xml_clean, predicted_clean))
            
            for i, line in enumerate(diff):
                if line.startswith('- ') or line.startswith('+ '):
                    char = line[2:]
                    if char not in [' ', '\t', '\n', '\r']:
                        change_type = "삭제됨" if line.startswith('-') else "추가됨"
                        context_start = max(0, i - 20)
                        context_end = min(len(diff), i + 20)
                        context = ''.join([d[2:] if d.startswith('  ') else d[2:] for d in diff[context_start:context_end]])
                        
                        difference_info = {
                            'position': i,
                            'type': change_type,
                            'character': char,
                            'context_before': context[:50],
                            'context_after': context[50:100]
                        }
                        non_space_differences.append(difference_info)
            
            # 모든 공백 외 차이점 출력 (처음 10개만 콘솔에, 전체는 보고서에)
            display_count = min(10, len(non_space_differences))
            for i in range(display_count):
                diff_info = non_space_differences[i]
                print(f"   {i+1}. [{diff_info['type']}] 위치 {diff_info['position']}: '{diff_info['character']}'")
                print(f"      컨텍스트: ...{diff_info['context_before']}[변경]{diff_info['context_after']}...")
            
            if len(non_space_differences) > 10:
                print(f"   ... 및 {len(non_space_differences) - 10}개 추가 차이점 (보고서에 전체 기록)")
        
        # 공백 차이점 카운트
        whitespace_diffs = len([c for c in difflib.ndiff(xml_text, predicted_text) 
                              if c.startswith('- ') or c.startswith('+ ')])
        if whitespace_diffs > 0:
            whitespace_count = whitespace_diffs
            print(f"📝 공백 차이점: {whitespace_diffs}개 (요약 생략)")
        
        return {
            'non_space_differences': non_space_differences,
            'whitespace_differences_count': whitespace_count,
            'xml_length': len(xml_text),
            'predicted_length': len(predicted_text),
            'has_content_differences': len(non_space_differences) > 0
        }
    
    def _log_matching_process(self, xml_items: List, result_items: List, 
                            matched_pairs: List, analysis_type: str = "PA"):
        """매칭 프로세스 로깅"""
        print(f"\n🔍 {analysis_type} 매칭 프로세스:")
        print(f"  XML 항목: {len(xml_items)}개")
        print(f"  결과 항목: {len(result_items)}개") 
        print(f"  매칭된 쌍: {len(matched_pairs)}개")
    
    def _get_integrity_summary_report(self) -> str:
        """무결성 분석 요약 보고서 생성"""
        summary = self.integrity_summary
        total = summary['processed_pairs']
        
        if total == 0:
            return "무결성 분석 데이터가 없습니다."
        
        report_lines = [
            "📊 콘텐츠 무결성 분석 결과",
            "=" * 50,
            f"총 처리된 쌍: {total}개",
            f"불일치 발견: {summary['total_mismatches']}개 ({summary['total_mismatches']/total*100:.1f}%)",
            "",
            "심각도별 분류:",
            f"  🔴 심각한 문제: {summary['severe_issues']}개",
            f"  🟡 중간 문제: {summary['moderate_issues']}개", 
            f"  🟢 경미한 문제: {summary['minor_issues']}개",
            ""
        ]
        
        if summary['total_mismatches'] > 0:
            report_lines.extend([
                "주요 불일치 사례 (최대 5개):",
                "-" * 30
            ])
            
            for issue in self.integrity_issues[:5]:
                severity_icon = {"severe": "🔴", "moderate": "🟡", "minor": "🟢"}[issue['severity']]
                report_lines.extend([
                    f"{severity_icon} {issue['pair_id']} (유사도: {issue['content_similarity']:.3f})",
                    f"  XML: {issue['xml_sample']}",
                    f"  결과: {issue['result_sample']}",
                    ""
                ])
        
        return "\n".join(report_lines)

    def _normalize_df_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame 컬럼 정규화"""
        if df is None or df.empty:
            return df
        
        col_map = {}
        cols = set(df.columns)
        
        # ID 컬럼 매핑
        for std, aliases in self._id_aliases.items():
            if std in cols:
                continue
            found = next((c for c in cols if c in aliases), None)
            if found:
                col_map[found] = std
        
        if col_map:
            df = df.rename(columns=col_map)
        
        return df

    def calculate_comprehensive_similarity(self, xml_file: str, 
                                        pa_result_file: str, 
                                        sa_result_file: str,
                                        xml_translation_file: str = None, 
                                        global_integrity: Dict = None) -> Dict[str, Any]:
        """종합 유사도 계산"""
        print("🎯 종합 XML 레벨 유사도 분석 시작...")
        
        # 전역 무결성 데이터 설정
        if global_integrity:
            self.global_integrity = {
                'original_similarity': global_integrity.get('global_source_text_similarity', 0),
                'translation_similarity': global_integrity.get('global_target_text_similarity', 0),
                'original_length_xml': global_integrity.get('global_source_len_gt', 0),
                'translation_length_xml': global_integrity.get('global_target_len_gt', 0),
                'original_length_predicted': global_integrity.get('global_source_len_pred', 0),
                'translation_length_predicted': global_integrity.get('global_target_len_pred', 0),
                'original_differences': {
                    'has_content_differences': global_integrity.get('global_source_ops_replace', 0) > 0,
                    'non_space_differences': []  # 임시로 빈 리스트
                },
                'translation_differences': {
                    'has_content_differences': global_integrity.get('global_target_ops_replace', 0) > 0,
                    'non_space_differences': []  # 임시로 빈 리스트
                }
            }
        
        results = {
            'xml_file': xml_file,
            'pa_result_file': pa_result_file,
            'sa_result_file': sa_result_file,
            'pa_analysis': {},
            'sa_analysis': {},
            'comprehensive_summary': {}
        }
        
        # PA 레벨 분석
        if pa_result_file and Path(pa_result_file).exists():
            results['pa_analysis'] = self.calculate_pa_similarity(xml_file, pa_result_file, xml_translation_file)
        elif pa_result_file:
            results['pa_analysis'] = {'error': 'PA 결과 파일이 존재하지 않습니다'}
        else:
            results['pa_analysis'] = {'error': 'PA 파일 없음'}
        
        # SA 레벨 분석 (번역문 XML 파일 포함)
        if Path(sa_result_file).exists():
            results['sa_analysis'] = self.calculate_sa_similarity(xml_file, sa_result_file, xml_translation_file)
        else:
            results['sa_analysis'] = {'error': 'SA 결과 파일이 존재하지 않습니다'}
        
        # 종합 요약 생성
        results['comprehensive_summary'] = self._generate_comprehensive_summary(results)
        
        # Integrity 분석 결과 추가
        if hasattr(self, 'integrity_summary') and self.integrity_summary['processed_pairs'] > 0:
            results['integrity_analysis'] = {
                'summary': self.integrity_summary.copy(),
                'detailed_issues': self.integrity_issues.copy(),
                'total_integrity_score': 1.0 - (self.integrity_summary['total_mismatches'] / max(1, self.integrity_summary['processed_pairs']))
            }
        
        # 종합 보고서 생성 (TXT/MD 형식)
        self._generate_comprehensive_report(results, xml_file, pa_result_file, sa_result_file, global_integrity)
        
        print("✅ 종합 XML 레벨 유사도 분석 완료")
        return results
    
    def _generate_comprehensive_summary(self, results: Dict) -> Dict:
        """종합 분석 요약 생성"""
        summary = {
            'pa_level': {},
            'sa_level': {},
            'overall_assessment': {}
        }
        
        # PA 레벨 요약
        pa_analysis = results.get('pa_analysis', {})
        if 'accuracy' in pa_analysis:
            summary['pa_level']['sentence_level_accuracy'] = pa_analysis['accuracy']
        
        # SA 레벨 요약
        sa_analysis = results.get('sa_analysis', {})
        if 'accuracy' in sa_analysis:
            summary['sa_level']['word_level_accuracy'] = sa_analysis['accuracy']
        
        # 전체 평가
        pa_score = summary['pa_level'].get('sentence_level_accuracy', 0)
        sa_score = summary['sa_level'].get('word_level_accuracy', 0)
        
        summary['overall_assessment'] = {
            'pa_sa_combined_score': (pa_score + sa_score) / 2,
            'processing_level_balance': abs(pa_score - sa_score),
            'recommendation': self._generate_recommendation(pa_score, sa_score)
        }
        
        return summary
    
    def _generate_recommendation(self, pa_score: float, sa_score: float) -> str:
        """점수 기반 개선 권장사항 생성"""
        if pa_score > 0.8 and sa_score > 0.8:
            return "매우 우수한 처리 품질입니다."
        elif pa_score < 0.5 and sa_score < 0.5:
            return "전반적인 처리 품질 개선이 필요합니다."
        elif pa_score > sa_score + 0.2:
            return "문장 레벨은 우수하나 구 레벨 정확도 개선이 필요합니다."
        elif sa_score > pa_score + 0.2:
            return "구 레벨은 우수하나 문장 레벨 정확도 개선이 필요합니다."
        else:
            return "균형 잡힌 처리 품질을 보여줍니다."
    
    def _group_sa_results_by_sentence(self, sa_df: pd.DataFrame, original_col: str, translation_col: str) -> List[Dict]:
        """SA 결과를 문장 단위로 그룹화 (원본 XML의 문장식별자 기반)"""
        sentence_groups = []
        
        # 문장식별자 컬럼 찾기
        sentence_id_col = None
        for col in sa_df.columns:
            if '문장식별자' in col or 'sentence_id' in col.lower():
                sentence_id_col = col
                break
        
        if not sentence_id_col:
            print("⚠️ SA 결과에서 문장식별자 컬럼을 찾을 수 없습니다. 순서 기반으로 그룹화합니다.")
            # 폴백: 순서 기반 그룹화
            return self._group_sa_results_by_order(sa_df, original_col, translation_col)
        
        # 문장식별자별로 그룹화
        grouped = sa_df.groupby(sentence_id_col)
        
        for sentence_id, group in grouped:
            phrases = []
            
            for idx, row in group.iterrows():
                original_text = str(row[original_col]) if pd.notna(row[original_col]) else ''
                translation_text = str(row[translation_col]) if pd.notna(row[translation_col]) else ''
                
                phrase_data = {
                    'idx': idx,
                    'original': original_text,  # 'text' 대신 'original' 사용
                    'translation': translation_text
                }
                phrases.append(phrase_data)
            
            if phrases:
                sentence_groups.append({
                    'sentence_id': str(sentence_id),  # XML과 매칭을 위해 문자열로 변환
                    'phrases': phrases,
                    'phrase_count': len(phrases)
                })
        
        # 문장 ID 순으로 정렬
        sentence_groups.sort(key=lambda x: x['sentence_id'])
        
        print(f"✅ SA 결과 문장별 그룹화 완료: {len(sentence_groups)}개 문장 (문장식별자 기반)")
        return sentence_groups
    
    def _group_sa_results_by_order(self, sa_df: pd.DataFrame, original_col: str, translation_col: str) -> List[Dict]:
        """SA 결과를 순서 기반으로 그룹화 (폴백 방식)"""
        sentence_groups = []
        current_sentence_phrases = []
        sentence_count = 0
        
        for idx, row in sa_df.iterrows():
            original_text = str(row[original_col]) if pd.notna(row[original_col]) else ''
            translation_text = str(row[translation_col]) if pd.notna(row[translation_col]) else ''
            
            phrase_data = {
                'idx': idx,
                'original': original_text,
                'translation': translation_text
            }
            
            current_sentence_phrases.append(phrase_data)
            
            # 문장 끝 판단 (문장부호 또는 일정 구 개수)
            if self._is_sentence_end(original_text, translation_text) or len(current_sentence_phrases) >= 10:
                if current_sentence_phrases:
                    sentence_count += 1
                    sentence_groups.append({
                        'sentence_id': f's_{sentence_count}',
                        'phrases': current_sentence_phrases,
                        'phrase_count': len(current_sentence_phrases)
                    })
                    current_sentence_phrases = []
        
        # 남은 구가 있으면 마지막 문장으로 추가
        if current_sentence_phrases:
            sentence_count += 1
            sentence_groups.append({
                'sentence_id': f's_{sentence_count}',
                'phrases': current_sentence_phrases,
                'phrase_count': len(current_sentence_phrases)
            })
        
        return sentence_groups

    def _is_sentence_end(self, original_text: str, translation_text: str) -> bool:
        """문장 끝인지 판단"""
        # 문장부호 확인
        sentence_endings = ['。', '．', '.', '?', '!', '？', '！', '다', '다.', '라', '라.']
        
        for ending in sentence_endings:
            if original_text.endswith(ending) or translation_text.endswith(ending):
                return True
        
        return False
    
    def _is_sentence_match(self, xml_group: Dict, sa_group: Dict) -> bool:
        """XML 문장 그룹과 SA 문장 그룹이 매칭되는지 판단 (문장식별자 기반)"""
        xml_sentence_id = xml_group['sentence_id']
        sa_sentence_id = sa_group['sentence_id']
        
        # 완전 일치 우선
        if xml_sentence_id == sa_sentence_id:
            return True
        
        # 문자열 정규화 후 비교 (예: 'jti_4c0201-[역주]당송팔대가문초한유1_원문_x-C2017-s_1' vs 's_1')
        xml_normalized = xml_sentence_id.split('-')[-1] if '-' in xml_sentence_id else xml_sentence_id
        sa_normalized = sa_sentence_id.split('-')[-1] if '-' in sa_sentence_id else sa_sentence_id
        
        return xml_normalized == sa_normalized

    def _generate_comprehensive_report(self, results: Dict, xml_file: str, pa_result_file: str, sa_result_file: str, global_integrity: Dict = None):
        """종합 분석 보고서 생성 (TXT 및 MD 형식)"""
        print("🔍 보고서 생성 함수 호출됨")
        try:
            # SA 결과 파일 경로를 기반으로 실제 결과 디렉토리 찾기
            sa_result_path = Path(sa_result_file)
            if "xml_pipeline_results" in str(sa_result_path):
                # XML 파이프라인 결과 디렉토리 구조 사용
                result_base_dir = sa_result_path.parent.parent  # sa_results/../ 로 올라가서 기본 디렉토리
                report_dir = result_base_dir / "accuracy"
            else:
                # 독립 실행시 기본 경로 사용
                xml_filename = Path(xml_file).stem
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                report_dir = Path("xml_pipeline_results") / f"{xml_filename}_{timestamp}" / "accuracy"
            
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # 보고서 내용 생성
            print("📝 보고서 내용 생성 중...")
            report_content = self._build_report_content(results, xml_file, pa_result_file, sa_result_file, global_integrity)
            print("✅ 보고서 내용 생성 완료")
            
            # TXT 보고서만 저장
            txt_report_path = report_dir / "comprehensive_similarity_report.txt"
            with open(txt_report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"📄 종합 보고서 생성 완료:")
            print(f"  📋 TXT: {txt_report_path}")
            
        except Exception as e:
            print(f"❌ 보고서 생성 오류: {e}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
    
    def _build_report_content(self, results: Dict, xml_file: str, pa_result_file: str, sa_result_file: str, global_integrity: Dict = None) -> str:
        """보고서 내용 구성"""
        
        content = []
        content.append("=" * 80)
        content.append("XML 유사도 분석 종합 보고서")
        content.append("=" * 80)
        content.append("")
        content.append(f"분석 일시: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}")
        content.append(f"XML 파일: {Path(xml_file).name}")
        content.append(f"PA 결과: {Path(pa_result_file).name}")
        content.append(f"SA 결과: {Path(sa_result_file).name}")
        content.append("")
        
        # PA 분석 결과
        content.append("📊 PA (Paragraph Analysis) 결과")
        content.append("-" * 50)
        pa_analysis = results.get('pa_analysis', {})
        # PA 분석이 성공했는지 확인 - xml_unit_count와 matched_pairs가 있고 0 이상이면 성공
        pa_has_data = (
            pa_analysis.get('xml_unit_count', 0) > 0 and 
            pa_analysis.get('matched_pairs', 0) >= 0 and
            'error' not in pa_analysis  # error 키가 없으면 성공
        ) or (
            # 또는 precision, recall, f1_score가 모두 있으면 성공
            pa_analysis.get('precision') is not None and 
            pa_analysis.get('recall') is not None and 
            pa_analysis.get('f1_score') is not None
        )
        
        if pa_has_data:
            content.append(f"• 분석 단위: 단락 레벨")
            content.append(f"• 매칭 방식: 번역문 기준 매칭")
            content.append(f"• XML 단위 수: {pa_analysis.get('xml_unit_count', 'N/A')}")
            content.append(f"• 결과 행 수: {pa_analysis.get('result_row_count', 'N/A')}")
            content.append(f"• 매칭된 쌍: {pa_analysis.get('matched_pairs', 'N/A')}")
            content.append(f"• Precision: {pa_analysis.get('precision', 0):.4f}")
            content.append(f"• Recall: {pa_analysis.get('recall', 0):.4f}")
            content.append(f"• F1 Score: {pa_analysis.get('f1_score', 0):.4f}")
            content.append("")
            content.append(f"  📈 유사도 세부 분석:")
            content.append(f"    - 평균 유사도: {pa_analysis.get('avg_similarity', 0):.4f}")
            content.append(f"    - 결합 유사도: {pa_analysis.get('avg_combined_similarity', pa_analysis.get('avg_similarity', 0)):.4f}")
        else:
            content.append(f"❌ 오류: {pa_analysis.get('error', 'PA 분석 데이터를 찾을 수 없습니다')}")
        content.append("")
        
        # SA 분석 결과
        content.append("📊 SA (Sentence Analysis) 결과")
        content.append("-" * 50)
        sa_analysis = results.get('sa_analysis', {})
        # error가 있어도 실제 데이터가 있으면 그것을 사용
        if sa_analysis.get('xml_unit_count') and sa_analysis.get('matched_pairs') is not None:
            content.append(f"• 분석 단위: 문장 레벨")
            content.append(f"• 매칭 방식: 원문 기준 매칭")
            content.append(f"• XML 단위 수: {sa_analysis.get('xml_unit_count', 'N/A')}")
            content.append(f"• 결과 행 수: {sa_analysis.get('result_row_count', 'N/A')}")
            content.append(f"• 매칭된 쌍: {sa_analysis.get('matched_pairs', 'N/A')}")
            content.append(f"• Precision: {sa_analysis.get('precision', 0):.4f}")
            content.append(f"• Recall: {sa_analysis.get('recall', 0):.4f}")
            content.append(f"• F1 Score: {sa_analysis.get('f1_score', 0):.4f}")
            content.append("")
            content.append("  📈 유사도 세부 분석:")
            content.append(f"    - 한 세트 유사도: {sa_analysis.get('avg_combined_similarity', sa_analysis.get('avg_similarity', 0)):.4f}")
            content.append(f"    - 원문만 유사도: {sa_analysis.get('avg_original_similarity', 0):.4f}")
            content.append(f"    - 번역문만 유사도: {sa_analysis.get('avg_translation_similarity', 0):.4f}")
        else:
            content.append(f"❌ 오류: {sa_analysis.get('error', 'SA 분석 데이터를 찾을 수 없습니다')}")
        content.append("")
        
        # 종합 비교
        content.append("🏆 PA vs SA 종합 비교")
        content.append("-" * 50)
        if 'error' not in pa_analysis and 'error' not in sa_analysis:
            pa_f1 = pa_analysis.get('f1_score', 0)
            sa_f1 = sa_analysis.get('f1_score', 0)
            pa_sim = pa_analysis.get('avg_combined_similarity', pa_analysis.get('avg_similarity', 0))
            sa_sim = sa_analysis.get('avg_combined_similarity', sa_analysis.get('avg_similarity', 0))
            
            content.append(f"• F1 Score 비교:")
            content.append(f"  - PA: {pa_f1:.4f}")
            content.append(f"  - SA: {sa_f1:.4f}")
            content.append(f"  - 차이: {abs(pa_f1 - sa_f1):.4f} ({'SA 우수' if sa_f1 > pa_f1 else 'PA 우수'})")
            content.append("")
            content.append(f"• 한 세트 유사도 비교:")
            content.append(f"  - PA: {pa_sim:.4f}")
            content.append(f"  - SA: {sa_sim:.4f}")
            content.append(f"  - 차이: {abs(pa_sim - sa_sim):.4f} ({'SA 높음' if sa_sim > pa_sim else 'PA 높음'})")
        content.append("")
        
        # 매칭 알고리즘 설명
        content.append("🔧 매칭 알고리즘")
        content.append("-" * 50)
        content.append("• PA: 번역문 기준 순차+스마트 매칭")
        content.append("  - 번역문을 기준으로 최적 매칭 찾기")
        content.append("  - 원문+번역문 한 세트 유사도 계산")
        content.append("")
        content.append("• SA: 원문 기준 순차+스마트 매칭")
        content.append("  - 원문을 기준으로 최적 매칭 찾기")
        content.append("  - 한 세트 + 원문/번역문 개별 유사도 모두 계산")
        content.append("")
        
        # 전역 무결성 분석 결과 추가
        if global_integrity and 'error' not in global_integrity:
            content.append("🔍 전역 무결성 분석")
            content.append("-" * 50)
            gi = global_integrity
            content.append(f"• 원문 전역 유사도: {gi.get('original_similarity', 0):.3f}")
            content.append(f"• 번역문 전역 유사도: {gi.get('translation_similarity', 0):.3f}")
            content.append(f"• XML 원문 길이: {gi.get('original_length_xml', 0):,}자")
            content.append(f"• XML 번역문 길이: {gi.get('translation_length_xml', 0):,}자")
            content.append(f"• 예측 원문 길이: {gi.get('original_length_predicted', 0):,}자")
            content.append(f"• 예측 번역문 길이: {gi.get('translation_length_predicted', 0):,}자")
            content.append("")
            
            # 원문 차이점 요약만
            orig_diff = gi.get('original_differences', {})
            if orig_diff.get('has_content_differences', False):
                content.append("⚠️ 원문 공백 외 차이점:")
                non_space_diffs = orig_diff.get('non_space_differences', [])
                content.append(f"  총 {len(non_space_diffs)}개 차이점 발견 (요약만 표시)")
                content.append("")
            else:
                content.append("✅ 원문: 공백 외 차이점 없음")
                if orig_diff.get('whitespace_differences_count', 0) > 0:
                    content.append(f"   공백 차이점: {orig_diff['whitespace_differences_count']}개")
                content.append("")
            
            # 번역문 차이점 요약만
            trans_diff = gi.get('translation_differences', {})
            if trans_diff.get('has_content_differences', False):
                content.append("⚠️ 번역문 공백 외 차이점:")
                non_space_diffs = trans_diff.get('non_space_differences', [])
                content.append(f"  총 {len(non_space_diffs)}개 차이점 발견 (요약만 표시)")
                content.append("")
            else:
                content.append("✅ 번역문: 공백 외 차이점 없음")
                if trans_diff.get('whitespace_differences_count', 0) > 0:
                    content.append(f"   공백 차이점: {trans_diff['whitespace_differences_count']}개")
                content.append("")
            
            # 무결성 평가
            orig_sim = gi.get('original_similarity', 0)
            trans_sim = gi.get('translation_similarity', 0)
            if orig_sim >= 0.95 and trans_sim >= 0.95:
                content.append("• 전역 무결성: ✅ 매우 우수")
            elif orig_sim >= 0.90 and trans_sim >= 0.90:
                content.append("• 전역 무결성: 🟡 양호")
            else:
                content.append("• 전역 무결성: ⚠️ 주의 필요")
            content.append("")
        
        content.append("=" * 80)
        
        return "\n".join(content)
    
    def _convert_to_markdown(self, text_content: str) -> str:
        """텍스트 내용을 마크다운 형식으로 변환"""
        lines = text_content.split('\n')
        md_lines = []
        
        for line in lines:
            if line.startswith('='):
                continue  # 구분선 제거
            elif line == "XML 유사도 분석 종합 보고서":
                md_lines.append("# XML 유사도 분석 종합 보고서")
            elif line.startswith('📊 PA') or line.startswith('📊 SA') or line.startswith('🏆') or line.startswith('🔧'):
                md_lines.append(f"## {line}")
            elif line.startswith('-' * 50):
                continue  # 소구분선 제거
            elif line.startswith('•'):
                md_lines.append(f"- {line[2:]}")  # 불릿 포인트
            elif line.startswith('  📈') or line.startswith('  -'):
                md_lines.append(f"  {line.strip()}")  # 들여쓰기 유지
            else:
                md_lines.append(line)
        
        return "\n".join(md_lines)


def main():
    """테스트 실행 - 원문+번역문 쌍으로 테스트"""
    calculator = XMLLevelSimilarityCalculator(use_embeddings=False)
    
    # 테스트 파일들 - 원문/번역문 쌍
    xml_original_file = "sources/jti_4c0201-[역주]당송팔대가문초한유1_원문_x-C2017.xml"
    xml_translation_file = "sources/jti_4c0201-[역주]당송팔대가문초한유1_번역문_x-C2017.xml"
    
    if Path(xml_original_file).exists() and Path(xml_translation_file).exists():
        print("🧪 XML 레벨별 유사도 계산 테스트 (원문+번역문)")
        
        # 개별 테스트
        sentences = calculator.xml_parser.extract_sentence_units(xml_original_file)
        words = calculator.xml_parser.extract_word_units(xml_original_file)
        
        print(f"\n📊 XML 단위 추출 결과:")
        print(f"  - 문장 단위: {len(sentences)}개")
        print(f"  - 어절 단위: {len(words)}개")
        
        if sentences:
            print(f"\n📝 문장 샘플:")
            for i, sent in enumerate(sentences[:3]):
                text = sent.get('text', '')[:100]
                print(f"  {i+1}. {text}...")
                
        # 실제 PA/SA 분석 테스트는 여기서 할 수 있음
        # pa_result = calculator.calculate_pa_similarity(xml_original_file, "pa/output.xlsx", xml_translation_file)
        # sa_result = calculator.calculate_sa_similarity(xml_original_file, "sa/output.xlsx", xml_translation_file)
    else:
        print(f"❌ 테스트 XML 파일이 존재하지 않습니다:")
        print(f"  원문: {xml_original_file}")
        print(f"  번역문: {xml_translation_file}")


if __name__ == "__main__":
    main()