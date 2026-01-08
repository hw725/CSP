"""
통합 한글 토씨 매칭 모듈 - Kiwipiepy 기반 (PA + SA 통합)
"""
import re
import logging
from typing import List, Dict, Tuple, Optional, Any

# Kiwipiepy import
try:
    from kiwipiepy import Kiwi
except ImportError:
    Kiwi = None
    logging.warning("Kiwipiepy를 찾을 수 없습니다. pip install kiwipiepy를 실행하세요.")

logger = logging.getLogger(__name__)

# 🆕 전역 싱글톤 인스턴스
_korean_particle_matcher_instance = None
_kiwi_tokenizer_instance = None
_kiwi_initialized = False

class KoreanParticleMatcher:
    """한글 토씨를 활용한 매칭 보조 클래스 (Kiwipiepy 기반 - PA/SA 통합)"""
    
    def __init__(self):
        # Kiwipiepy 토크나이저는 전역적으로 관리
        pass
        
        # 주요 한글 토씨 패턴들 (고전 한문 번역체 포함)
        self.particles = {
            # 주격 조사
            '주격': ['이', '가', '이면', '가면', '이여', '이요', '이로되', '이나'],
            # 목적격 조사  
            '목적격': ['을', '를', '로되', '을지언정'],
            # 처소격 조사
            '처소격': ['에', '에서', '으로', '로', '에게', '한테', '에서부터', '로부터', '로써', '으로써'],
            # 연결어미 (고어 포함)
            '연결': ['하고', '하며', '하여', '고', '며', '하되', '하나', '하니', '하는도다', '함이', '함에'],
            # 종결어미 (고어 포함)
            '종결': [
                # 현대어
                '라', '다', '요', '니다', '하다', '하니', '하리',
                # 고어 종결어미
                '도다', '노라', '로다', '이로다', '이여', '이니라', '하도다', 
                '하노라', '하로다', '하니라', '하리라', '하겠노라',
                # 감탄형
                '이로구나', '하는구나', '하는도다', '아', '어', '이여',
                # 의문형 고어
                '인가', '하는가', '하리요', '이리요',
                # 명령형 고어
                '하라', '할지어다', '하지어다'
            ],
            # 관형격
            '관형격': ['의', '은', '는', '인', '한', '할', '함의', '하는'],
            # 부사격
            '부사격': ['와', '과', '랑', '이랑', '하고', '하며'],
            # 🆕 고어 전용 어미들
            '고어_서술': [
                '이라', '이니라', '이로다', '라', '로다', 
                '하니라', '하도다', '하노라', '하로다'
            ],
            '고어_관형': [
                '인', '하는', '할', '한', '하던', '하신', '함', '함이'
            ],
            '고어_부사': [
                '이', '히', '게', '도록', '토록', '하여', '하되'
            ],
            '고어_감탄': [
                '이여', '아', '어', '이로구나', '하는구나', '이로다'
            ]
        }
        
        # 토씨별 가중치 (매칭 힌트로서의 중요도) - 고어 가중치 추가
        self.particle_weights = {
            '목적격': 0.8,      # 목적어는 매칭에 중요한 힌트
            '처소격': 0.7,      # 처소도 중요한 힌트
            '주격': 0.6,        # 주어도 도움됨
            '연결': 0.5,        # 연결어미는 구조 파악에 도움
            '종결': 0.4,        # 종결어미는 문장 끝 표시
            '관형격': 0.3,      # 관형격은 보조적
            '부사격': 0.3,      # 부사격도 보조적
            # 고어 어미 가중치 (고전 번역에서 중요)
            '고어_서술': 0.6,   # 고어 서술어미는 중요
            '고어_관형': 0.4,   # 고어 관형어미
            '고어_부사': 0.3,   # 고어 부사어미
            '고어_감탄': 0.5    # 고어 감탄어미
        }
        
        # 🆕 고어 패턴 인식용 프리미엄 어미들
        self.premium_archaic_endings = ['이니라', '하니라', '도다', '노라', '로다']
        self.classical_patterns = {
            'strong': ['이니라', '하니라', '도다', '노라', '로다', '이로다'],
            'medium': ['하여', '하되', '하나', '하리', '이요', '이여'],
            'weak': ['함이', '함에', '인가', '하는가', '할지어다']
        }
    
    def _get_kiwi_tokenizer(self):
        """Kiwi 토크나이저 지연 초기화 (전역 싱글톤)"""
        global _kiwi_tokenizer_instance, _kiwi_initialized
        
        if _kiwi_tokenizer_instance is None and not _kiwi_initialized:
            try:
                if Kiwi is None:
                    raise ImportError("Kiwipiepy가 설치되지 않았습니다")
                
                # Kiwipiepy 직접 초기화 (전역적으로 한 번만)
                _kiwi_tokenizer_instance = Kiwi()
                _kiwi_initialized = True
                logger.info("Kiwipiepy 토크나이저 초기화 완료")
            except Exception as e:
                logger.warning(f"Kiwipiepy 초기화 실패, 폴백 모드 사용: {e}")
                _kiwi_tokenizer_instance = None
                _kiwi_initialized = True  # 실패해도 다시 시도하지 않음
        
        return _kiwi_tokenizer_instance
    
    def _map_pos_to_category(self, pos_tag: str, form: str) -> str:
        """Kiwipiepy 품사 태그를 우리 카테고리로 매핑"""
        # 조사 태그 매핑
        if pos_tag == 'JKS':  # 주격조사
            return '주격'
        elif pos_tag == 'JKO':  # 목적격조사
            return '목적격'
        elif pos_tag in ['JKG', 'JKB']:  # 관형격조사, 부사격조사
            return '관형격' if pos_tag == 'JKG' else '부사격'
        elif pos_tag in ['JKV', 'JKC']:  # 호격조사, 보격조사
            return '처소격'
        elif pos_tag in ['JX', 'JC']:  # 보조사, 접속조사
            return '부사격'
        else:
            return '기타_조사'
    
    def _map_ending_to_category(self, pos_tag: str, form: str) -> Optional[str]:
        """Kiwipiepy 어미 태그를 우리 카테고리로 매핑 (고어 감지 포함)"""
        
        # 🆕 고어 어미 우선 확인 (Kiwipiepy가 인식한 형태 기준)
        if form in ['이니라', '하니라', '도다', '노라', '로다', '이로다', '하도다']:
            return '고어_서술'
        elif form in ['이여', '아', '어', '이로구나', '하는구나']:
            return '고어_감탄'
        elif form in ['하여', '하되', '하나', '하니']:
            if pos_tag == 'EC':  # 연결어미
                return '연결'
            else:
                return '고어_부사'
        
        # 일반 어미 매핑
        if pos_tag == 'EF':  # 종결어미
            return '종결'
        elif pos_tag == 'EC':  # 연결어미
            return '연결'
        elif pos_tag in ['ETN', 'ETM']:  # 전성어미
            return '관형격'
        elif pos_tag == 'EP':  # 선어말어미
            return None  # 선어말어미는 제외
        else:
            return '기타_어미'
    
    def extract_particles_from_text(self, text: str) -> List[Tuple[str, str, int]]:
        """
        텍스트에서 한글 토씨 추출 (Kiwipiepy 기반 하이브리드 토크나이저)
        
        Returns:
            List[Tuple[str, str, int]]: (토씨, 카테고리, 위치) 목록
        """
        particles_found = []
        
        # Kiwipiepy로 한글 분석 (하이브리드 토크나이저의 핵심)
        kiwi = self._get_kiwi_tokenizer()
        if kiwi:
            try:
                tokens = kiwi.analyze(text)
                if tokens and len(tokens) > 0 and len(tokens[0]) > 0:
                    char_pos = 0
                    
                    for token in tokens[0][0]:
                        form = token.form
                        pos = token.tag
                        
                        # 조사 처리
                        if pos.startswith('JK') or pos.startswith('JX') or pos.startswith('JC'):
                            category = self._map_pos_to_category(pos, form)
                            particles_found.append((form, category, char_pos))
                        
                        # 어미 처리 (고어 인식 포함)
                        elif pos.startswith('E'):
                            category = self._map_ending_to_category(pos, form)
                            if category:
                                particles_found.append((form, category, char_pos))
                        
                        char_pos += len(form)
                    
                    particles_found.sort(key=lambda x: x[2])
                    return particles_found
                else:
                    logger.warning(f"Kiwipiepy 분석 결과가 비어있음: {text}")
                    
            except Exception as e:
                logger.warning(f"Kiwipiepy 분석 실패: {e}")
        
        # 폴백: 패턴 매칭으로 토씨 찾기
        for category, particle_list in self.particles.items():
            for particle in particle_list:
                pos = text.find(particle)
                if pos != -1:
                    particles_found.append((particle, category, pos))
        
        # 위치순으로 정렬
        particles_found.sort(key=lambda x: x[2])
        return particles_found

    def calculate_particle_similarity(self, src_particles: List[Tuple[str, str, int]], 
                                    tgt_particles: List[Tuple[str, str, int]]) -> float:
        """
        원문과 번역문의 토씨 패턴 유사도 계산
        
        Args:
            src_particles: 원문 토씨 목록
            tgt_particles: 번역문 토씨 목록
            
        Returns:
            float: 0.0-1.0 사이의 유사도 점수
        """
        if not src_particles and not tgt_particles:
            return 1.0
        
        if not src_particles or not tgt_particles:
            return 0.0
        
        # 카테고리별 매칭 점수 계산
        src_categories = [p[1] for p in src_particles]
        tgt_categories = [p[1] for p in tgt_particles]
        
        matched_categories = set(src_categories) & set(tgt_categories)
        total_categories = set(src_categories) | set(tgt_categories)
        
        if not total_categories:
            return 1.0
        
        # 기본 유사도: 공통 카테고리 비율
        base_similarity = len(matched_categories) / len(total_categories)
        
        # 가중치 적용: 중요한 토씨일수록 높은 점수
        weighted_score = 0.0
        total_weight = 0.0
        
        for category in matched_categories:
            weight = self.particle_weights.get(category, 0.1)
            weighted_score += weight
            total_weight += weight
        
        for category in total_categories - matched_categories:
            weight = self.particle_weights.get(category, 0.1)
            total_weight += weight
        
        if total_weight > 0:
            weighted_similarity = weighted_score / total_weight
        else:
            weighted_similarity = 0.0
        
        # 기본 유사도와 가중치 유사도의 조합
        final_similarity = (base_similarity * 0.4) + (weighted_similarity * 0.6)
        
        return min(1.0, final_similarity)
    
    # ============ 고어 패턴 감지 메서드들 ============
    
    def detect_archaic_patterns(self, text: str, mode: str = 'SA') -> Dict[str, Any]:
        """고어 패턴 감지 (Kiwipiepy 결과 기반)"""
        particles = self.extract_particles_from_text(text)
        
        strong_count = medium_count = weak_count = 0
        patterns_found = []
        
        for form, category, pos in particles:
            if category.startswith('고어_'):
                if form in self.classical_patterns['strong']:
                    strong_count += 1
                    patterns_found.append(('strong', form))
                elif form in self.classical_patterns['medium']:
                    medium_count += 1
                    patterns_found.append(('medium', form))
                elif form in self.classical_patterns['weak']:
                    weak_count += 1
                    patterns_found.append(('weak', form))
        
        total_patterns = strong_count + medium_count + weak_count
        text_length = len(text.replace(' ', ''))
        pattern_density = total_patterns / text_length if text_length > 0 else 0.0
        
        # 구조적 보너스 계산
        structural_bonus = 0.0
        if strong_count > 0:
            structural_bonus += min(0.3, strong_count * 0.15)
        if medium_count > 0:
            structural_bonus += min(0.15, medium_count * 0.08)
        if weak_count > 0:
            structural_bonus += min(0.1, weak_count * 0.05)
        
        # 신뢰도 판정
        if structural_bonus >= 0.2:
            confidence = 'high'
        elif structural_bonus >= 0.1:
            confidence = 'medium'
        elif structural_bonus >= 0.05:
            confidence = 'low'
        else:
            confidence = 'none'
        
        return {
            'patterns_found': patterns_found,
            'strong_count': strong_count,
            'medium_count': medium_count,
            'weak_count': weak_count,
            'total_patterns': total_patterns,
            'pattern_density': pattern_density,
            'structural_bonus': structural_bonus,
            'confidence': confidence,
            'text_length': text_length
        }
    
    def get_archaic_bonus(self, text: str, mode: str = 'SA') -> float:
        """고어 구조적 보너스 점수 반환"""
        result = self.detect_archaic_patterns(text, mode)
        return result['structural_bonus']
    
    def is_archaic_translation(self, text: str) -> bool:
        """텍스트가 고어체 번역인지 판별"""
        result = self.detect_archaic_patterns(text, mode='PA')
        return result['confidence'] in ['high', 'medium']
    
    # ============ PA용 메서드들 ============
    
    def enhance_alignment_with_particles(self, alignments: List[Dict]) -> List[Dict]:
        """
        기존 정렬 결과에 한글 토씨 정보를 추가하여 보완 (PA용)
        
        Args:
            alignments: 기존 PA 정렬 결과
            
        Returns:
            List[Dict]: 토씨 정보가 추가된 정렬 결과
        """
        enhanced_alignments = []
        
        for alignment in alignments:
            src_text = alignment.get('원문', '')
            tgt_text = alignment.get('번역문', '')
            
            # 토씨 추출
            src_particles = self.extract_particles_from_text(src_text)
            tgt_particles = self.extract_particles_from_text(tgt_text)
            
            # 토씨 기반 유사도 계산
            particle_similarity = self.calculate_particle_similarity(src_particles, tgt_particles)
            
            # 🆕 고어 패턴 보정 추가
            archaic_bonus = self.get_archaic_bonus(tgt_text, mode='PA')
            
            # 기존 유사도와 토씨 유사도 결합 (기존 우선, 토씨는 보정)
            original_similarity = alignment.get('similarity', 0.0)
            
            # 토씨 유사도가 높으면 기존 유사도를 약간 향상시킴 (최대 +0.1)
            # 🆕 고어 패턴 보너스도 추가 (최대 +0.3)
            enhanced_similarity = original_similarity + (particle_similarity * 0.1) + archaic_bonus
            enhanced_similarity = min(1.0, enhanced_similarity)
            
            # 토씨 정보 추가
            enhanced_alignment = alignment.copy()
            enhanced_alignment.update({
                'similarity': enhanced_similarity,
                'original_similarity': original_similarity,
                'particle_similarity': particle_similarity,
                'archaic_bonus': archaic_bonus,  # 🆕 고어 보너스 정보 추가
                'src_particles': src_particles,
                'tgt_particles': tgt_particles,
                'particle_boost': particle_similarity * 0.1,
                'archaic_boost': archaic_bonus  # 🆕 고어 부스트 정보 추가
            })
            
            enhanced_alignments.append(enhanced_alignment)
        
        return enhanced_alignments
    
    # ============ SA용 메서드들 ============
    
    def _calculate_embedding_similarity(self, src_unit: str, tgt_unit: str) -> float:
        """임베딩 유사도 계산 (SA용)"""
        try:
            # 실제 BGE 임베딩을 사용한 유사도 계산
            from common.embedders import get_embedder
            embedder = get_embedder('bge')
            
            # 임베딩 계산
            src_embedding = embedder([src_unit], batch_size=1)[0]
            tgt_embedding = embedder([tgt_unit], batch_size=1)[0]
            
            # 코사인 유사도 계산
            import numpy as np
            dot_product = np.dot(src_embedding, tgt_embedding)
            norm_src = np.linalg.norm(src_embedding)
            norm_tgt = np.linalg.norm(tgt_embedding)
            
            if norm_src == 0 or norm_tgt == 0:
                return 0.0
            
            similarity = dot_product / (norm_src * norm_tgt)
            return max(0.0, min(1.0, similarity))
                
        except Exception as e:
            logger.warning(f"임베딩 유사도 계산 실패: {e}")
            # 폴백: 간단한 문자열 유사도
            from difflib import SequenceMatcher
            return SequenceMatcher(None, src_unit, tgt_unit).ratio()
    
    def _assess_match_quality(self, similarity: float) -> str:
        """토씨 유사도 기반 매칭 품질 평가"""
        if similarity >= 0.8:
            return 'excellent'
        elif similarity >= 0.6:
            return 'good'
        elif similarity >= 0.4:
            return 'fair'
        elif similarity >= 0.2:
            return 'poor'
        else:
            return 'very_poor'
    
    def enhance_sa_single_units(self, src_unit: str, tgt_unit: str) -> Dict[str, Any]:
        """
        SA의 개별 단위 쌍에 대한 토씨 매칭 정보 추가 (Kiwipiepy 기반)
        
        Args:
            src_unit: 원문 단위 (공백으로 분할된 하나의 단위)
            tgt_unit: 번역문 단위 (정렬된 하나의 단위)
            
        Returns:
            토씨 매칭 정보가 포함된 딕셔너리
        """
        try:
            # Kiwipiepy로 토씨 추출 (원문+번역문 모두)
            src_particles = self.extract_particles_from_text(src_unit)
            tgt_particles = self.extract_particles_from_text(tgt_unit)
            
            # 토씨 유사도 계산
            particle_similarity = self.calculate_particle_similarity(
                src_particles, tgt_particles
            )
            
            # 임베딩 유사도 계산 (SA DP와 동일한 방식)
            embedding_similarity = self._calculate_embedding_similarity(src_unit, tgt_unit)
            
            # 최종 유사도: 임베딩 + 토씨 보정
            final_similarity = embedding_similarity + (particle_similarity * 0.15)
            final_similarity = min(1.0, final_similarity)
            
            return {
                'embedding_similarity': embedding_similarity,
                'particle_similarity': particle_similarity,
                'final_similarity': final_similarity,
                'particle_boost': particle_similarity * 0.15,
                'src_particles': [p[0] for p in src_particles],  # 토씨만 저장
                'tgt_particles': [p[0] for p in tgt_particles], 
                'src_particle_categories': [p[1] for p in src_particles],
                'tgt_particle_categories': [p[1] for p in tgt_particles],
                'particle_match_quality': self._assess_match_quality(particle_similarity),
                'tokenizer_used': "kiwipiepy",
                'kiwipiepy_analysis': True
            }
            
        except Exception as e:
            logger.warning(f"SA 토씨 매칭 실패: {e}")
            return {
                'embedding_similarity': 0.0,
                'particle_similarity': 0.0,
                'final_similarity': 0.0,
                'particle_boost': 0.0,
                'src_particles': [],
                'tgt_particles': [],
                'src_particle_categories': [],
                'tgt_particle_categories': [],
                'particle_match_quality': 'unknown',
                'tokenizer_used': 'error',
                'kiwipiepy_analysis': False
            }
    
    def enhance_sa_row_result(self, row_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        SA 행 결과에 토씨 정보 추가
        
        Args:
            row_result: SA process_single_row의 결과
            
        Returns:
            토씨 정보가 추가된 행 결과
        """
        try:
            src_text = row_result.get('원문', '')
            tgt_text = row_result.get('번역문', '')
            
            # 전체 문장의 토씨 분석
            src_particles = self.extract_particles_from_text(src_text)
            tgt_particles = self.extract_particles_from_text(tgt_text)
            
            # 토씨 유사도 계산
            particle_similarity = self.calculate_particle_similarity(src_particles, tgt_particles)
            
            # 고어 패턴 분석
            archaic_analysis = self.detect_archaic_patterns(tgt_text, mode='SA')
            
            # 결과에 토씨 정보 추가
            enhanced_result = row_result.copy()
            enhanced_result.update({
                'particle_similarity': particle_similarity,
                'src_particles_count': len(src_particles),
                'tgt_particles_count': len(tgt_particles),
                'archaic_score': archaic_analysis['structural_bonus'],
                'archaic_confidence': archaic_analysis['confidence'],
                'archaic_patterns_found': archaic_analysis['patterns_found'],
                'particle_analysis': {
                    'src_particles': src_particles,
                    'tgt_particles': tgt_particles,
                    'similarity': particle_similarity,
                    'archaic_analysis': archaic_analysis
                }
            })
            
            return enhanced_result
            
        except Exception as e:
            logger.warning(f"SA 행 결과 토씨 분석 실패: {e}")
            return row_result
            
            # 토씨 매칭 정보 추가
            particle_info = self.enhance_sa_single_units(src_text, tgt_text)
            
            # 기존 결과에 토씨 정보 추가
            enhanced_result = row_result.copy()
            enhanced_result.update(particle_info)
            
            return enhanced_result
            
        except Exception as e:
            logger.warning(f"SA 행 토씨 매칭 실패: {e}")
            return row_result
    
    def enhance_sa_rows_batch(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        SA 행들의 배치에 토씨 정보 추가
        
        Args:
            rows: SA process_single_row의 결과 리스트
            
        Returns:
            토씨 정보가 추가된 행 리스트
        """
        enhanced_rows = []
        
        for row in rows:
            enhanced_row = self.enhance_sa_row_result(row)
            enhanced_rows.append(enhanced_row)
        
        return enhanced_rows
    
    # ============ 공통 분석 메서드들 ============
    
    def analyze_particle_patterns(self, text: str) -> Dict[str, any]:
        """
        텍스트의 토씨 패턴 분석 (디버깅/분석용) - 고어 어미 특별 분석 포함
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            Dict: 토씨 패턴 분석 결과
        """
        particles = self.extract_particles_from_text(text)
        
        analysis = {
            'total_particles': len(particles),
            'particles_by_category': {},
            'particle_sequence': [p[0] for p in particles],
            'category_sequence': [p[1] for p in particles],
            'detailed_particles': particles,
            # 🆕 고어 어미 특별 분석
            'archaic_analysis': {
                'has_archaic_endings': False,
                'archaic_count': 0,
                'archaic_types': {},
                'translation_style': 'unknown'
            }
        }
        
        # 카테고리별 집계
        archaic_count = 0
        for particle, category, pos in particles:
            if category not in analysis['particles_by_category']:
                analysis['particles_by_category'][category] = []
            analysis['particles_by_category'][category].append(particle)
            
            # 고어 패턴 카운트
            if category.startswith('고어_'):
                archaic_count += 1
                if category not in analysis['archaic_analysis']['archaic_types']:
                    analysis['archaic_analysis']['archaic_types'][category] = []
                analysis['archaic_analysis']['archaic_types'][category].append(particle)
        
        # 고어 어미 종합 분석
        analysis['archaic_analysis']['has_archaic_endings'] = archaic_count > 0
        analysis['archaic_analysis']['archaic_count'] = archaic_count
        
        # 번역 스타일 판정
        if archaic_count >= 3:
            analysis['archaic_analysis']['translation_style'] = 'classical'
        elif archaic_count >= 1:
            analysis['archaic_analysis']['translation_style'] = 'mixed'
        else:
            analysis['archaic_analysis']['translation_style'] = 'modern'
            
        return analysis
    
    def is_archaic_translation(self, text: str) -> bool:
        """
        텍스트가 고어체 번역인지 판별 (common 모듈 활용)
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            bool: 고어체 번역 여부
        """
        analysis = self.analyze_particle_patterns(text)
        archaic_analysis = analysis['archaic_analysis']
        
        return archaic_analysis['translation_style'] in ['classical', 'mixed']
    
    def get_archaic_score(self, text: str) -> float:
        """
        텍스트의 고어 점수 반환 (0.0-1.0)
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            float: 고어 점수 (높을수록 고어체)
        """
        archaic_analysis = self.detect_archaic_patterns(text, mode='SA')
        return archaic_analysis['structural_bonus']


# 🆕 싱글톤 접근 함수
def get_korean_particle_matcher():
    """싱글톤 KoreanParticleMatcher 인스턴스 반환 (재사용)"""
    global _korean_particle_matcher_instance
    if _korean_particle_matcher_instance is None:
        _korean_particle_matcher_instance = KoreanParticleMatcher()
    return _korean_particle_matcher_instance

# ============ PA용 외부 함수들 ============

def enhance_pa_alignments_with_particles(alignments: List[Dict]) -> List[Dict]:
    """
    PA 정렬 결과에 한글 토씨 힌트 추가 (기존 로직은 보존)
    
    Args:
        alignments: 기존 PA 정렬 결과
        
    Returns:
        List[Dict]: 토씨 힌트가 추가된 정렬 결과
    """
    matcher = get_korean_particle_matcher()
    return matcher.enhance_alignment_with_particles(alignments)

# ============ SA용 외부 함수들 ============

def enhance_sa_results_with_particles(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    SA 결과에 한글 토씨 정보 추가 (외부 호출용)
    
    Args:
        rows: SA process_single_row의 결과 리스트
        
    Returns:
        토씨 정보가 추가된 행 리스트
    """
    matcher = get_korean_particle_matcher()
    return matcher.enhance_sa_rows_batch(rows)

def analyze_sa_unit_particles(src_unit: str, tgt_unit: str) -> Dict[str, Any]:
    """
    SA 단위 쌍의 토씨 분석 (외부 호출용)
    """
    matcher = get_korean_particle_matcher()
    return matcher.enhance_sa_single_units(src_unit, tgt_unit)

# ============ 공통 외부 함수들 ============

def analyze_text_particles(text: str) -> Dict[str, any]:
    """
    텍스트의 토씨 패턴 분석 (외부 호출용)
    """
    matcher = get_korean_particle_matcher()
    return matcher.analyze_particle_patterns(text)

def check_archaic_translation(text: str) -> bool:
    """
    고어체 번역 여부 확인 (외부 호출용 - common 모듈 활용)
    """
    matcher = get_korean_particle_matcher()
    result = matcher.detect_archaic_patterns(text, mode='PA')
    return result['confidence'] in ['high', 'medium']

def get_archaic_bonus(text: str, mode: str = 'SA') -> float:
    """고어 구조적 보너스 점수 반환 (외부 호출용)"""
    matcher = get_korean_particle_matcher()
    return matcher.get_archaic_bonus(text, mode)

def detect_archaic_patterns(text: str, mode: str = 'SA') -> Dict[str, Any]:
    """고어 패턴 감지 (외부 호출용)"""
    matcher = get_korean_particle_matcher()
    return matcher.detect_archaic_patterns(text, mode)


# 싱글톤 인스턴스 생성 (재사용)
korean_particle_matcher = get_korean_particle_matcher()
