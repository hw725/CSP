"""
XML 파이프라인용 최적화 설정 통합 패치

기존 xml_pipeline_processor.py의 PA/SA 호출 부분에 최적화 설정을 적용하는 헬퍼 함수들
"""

import sys
import re
from pathlib import Path
from typing import Optional, Dict, List

# 성능 최적화기 import
try:
    from .performance_optimizer import PerformanceOptimizer, get_optimized_config_for_jti
    from .docker_performance_optimizer import DockerPerformanceOptimizer, create_docker_optimizer
except ImportError:
    # 성능 최적화기가 없는 경우 기본 설정 사용
    class PerformanceOptimizer:
        def optimize_pa_command(self, cmd): return cmd
        def optimize_sa_command(self, cmd): return cmd
        def setup_environment_optimization(self): pass
    
    class DockerPerformanceOptimizer(PerformanceOptimizer):
        pass
    
    def get_optimized_config_for_jti(jti_code): 
        return None
    
    def create_docker_optimizer():
        return PerformanceOptimizer()

# JTI 코드 매핑 import
try:
    from .jti_code_mappings import (
        get_jti_by_text_name, 
        get_text_name_by_jti, 
        validate_jti_code,
        TEXT_TO_JTI_MAPPINGS
    )
except ImportError:
    # JTI 매핑이 없는 경우 더미 함수
    def get_jti_by_text_name(text_name: str) -> Optional[str]:
        return None
    def get_text_name_by_jti(jti_code: str) -> str:
        return f"Unknown_{jti_code}"
    def validate_jti_code(jti_code: str) -> bool:
        return True
    TEXT_TO_JTI_MAPPINGS = {}

# 최적화 설정 import
try:
    from config.text_optimization_configs import get_pa_config_by_jti, get_sa_config_by_jti
except ImportError:
    # config 모듈이 없는 경우 더미 함수
    def get_pa_config_by_jti(jti_code: str) -> Optional[Dict]:
        return None
    def get_sa_config_by_jti(jti_code: str) -> Optional[Dict]:
        return None

# 글로벌 성능 최적화기 인스턴스
_performance_optimizer = None

def get_performance_optimizer():
    """성능 최적화기 인스턴스 반환 (싱글톤) - Docker 환경 자동 감지"""
    global _performance_optimizer
    if _performance_optimizer is None:
        # Docker 환경 감지하여 적절한 최적화기 선택
        try:
            docker_optimizer = create_docker_optimizer()
            if hasattr(docker_optimizer, 'is_in_container') and docker_optimizer.is_in_container:
                print("🐳 Docker 환경 감지됨 - Docker 전용 최적화기 사용")
                _performance_optimizer = docker_optimizer
            else:
                print("💻 로컬 환경 - 기본 최적화기 사용")
                _performance_optimizer = PerformanceOptimizer()
        except Exception as e:
            print(f"⚠️ Docker 최적화기 로딩 실패, 기본 최적화기 사용: {e}")
            _performance_optimizer = PerformanceOptimizer()
        
        _performance_optimizer.setup_environment_optimization()
    return _performance_optimizer

def extract_jti_from_xml_pair(xml_pair) -> Optional[str]:
    """XMLPair 객체에서 JTI 코드 추출 (실제 데이터 기반)"""
    try:
        # pair_id나 name에서 JTI 코드 찾기
        sources = [
            getattr(xml_pair, 'pair_id', ''),
            getattr(xml_pair, 'id', ''),
            getattr(xml_pair, 'name', ''),
            str(getattr(xml_pair, 'original_path', '')),
            str(getattr(xml_pair, 'translation_path', ''))
        ]
        
        for source in sources:
            if not source:
                continue
                
            source_lower = str(source).lower()
            
            # 1. JTI 코드 패턴 직접 추출 (우선순위 높음)
            jti_patterns = [
                r'jti_([0-9][a-z][0-9]{4})',  # jti_1h0301 형식
                r'([0-9][a-z][0-9]{4})',      # 1h0301 형식 (단독)
            ]
            
            for pattern in jti_patterns:
                match = re.search(pattern, source_lower)
                if match:
                    code = match.group(1) if pattern.startswith('jti_') else match.group(1)
                    # 실제 처리된 JTI 코드인지 검증
                    if validate_jti_code(code):
                        print(f"🎯 JTI 코드 직접 추출: '{source}' → {code}")
                        return code
        
        # 2. 텍스트명으로 JTI 코드 매핑 (실제 데이터 기반)
        for source in sources:
            if not source:
                continue
                
            source_str = str(source)
            
            # 실제 config 기반 텍스트명 매핑 사용
            jti_code = get_jti_by_text_name(source_str)
            if jti_code:
                text_name = get_text_name_by_jti(jti_code)
                print(f"🎯 텍스트명 매핑 감지: '{source_str}' → JTI {jti_code} ({text_name})")
                return jti_code
                    
        return None
        
    except Exception as e:
        print(f"⚠️ JTI 코드 추출 실패: {e}")
        return None

def build_optimized_pa_cmd(xml_pair, input_file: str, output_file: str, base_cmd: List[str]) -> List[str]:
    """최적화된 PA 명령어 생성 (성능 최적화 포함)"""
    
    # JTI 코드 추출
    jti_code = extract_jti_from_xml_pair(xml_pair)
    
    # 성능 최적화기 적용
    optimizer = get_performance_optimizer()
    
    if not jti_code:
        # JTI 코드가 없어도 기본 성능 최적화는 적용
        return optimizer.optimize_pa_command(base_cmd)
    
    # 최적화 설정 로드
    config = get_pa_config_by_jti(jti_code)
    
    # 기본 명령어에서 시작
    cmd = base_cmd.copy()
    
    # 성능 최적화 먼저 적용
    cmd = optimizer.optimize_pa_command(cmd)
    
    # PA 최적화 설정 적용
    pa_config = config.get('pa_config', {}) if config else {}
    
    # 실제 텍스트명 가져오기
    actual_text_name = get_text_name_by_jti(jti_code)
    target_config_name = config.get('text_name', '알 수 없음') if config else '설정 없음'
    
    print(f"🎯 PA 최적화 적용: JTI {jti_code} → {actual_text_name}")
    if config:
        print(f"   성능 등급: {config.get('performance_grade', 'N/A')}, 목표 F1: {config.get('target_f1', 'N/A')}")
    else:
        print(f"   ⚠️ 최적화 설정 없음 - 기본 성능 최적화만 적용")
    
    # 파라미터 적용
    if 'max_length' in pa_config:
        # 기존 --max-length 제거 후 새 값 추가
        cmd = [x for x in cmd if not x.startswith('--max-length')]
        cmd.extend(['--max-length', str(pa_config['max_length'])])
    
    if 'threshold' in pa_config:
        cmd = [x for x in cmd if not x.startswith('--threshold')]
        cmd.extend(['--threshold', str(pa_config['threshold'])])
        
    if 'max_workers' in pa_config:
        # 기존 --max-workers 값 교체
        try:
            idx = cmd.index('--max-workers')
            cmd[idx + 1] = str(pa_config['max_workers'])
        except ValueError:
            cmd.extend(['--max-workers', str(pa_config['max_workers'])])
    
    if 'batch_size' in pa_config:
        cmd.extend(['--batch-size', str(pa_config['batch_size'])])
    
    print(f"   PA 최적화 파라미터 {len(pa_config)}개 적용됨")
    
    return cmd

def build_optimized_sa_cmd(xml_pair, input_file: str, output_file: str, base_cmd: List[str]) -> List[str]:
    """최적화된 SA 명령어 생성 (성능 최적화 포함)"""
    
    # JTI 코드 추출
    jti_code = extract_jti_from_xml_pair(xml_pair)
    
    # 성능 최적화기 적용
    optimizer = get_performance_optimizer()
    
    if not jti_code:
        # JTI 코드가 없어도 기본 성능 최적화는 적용
        return optimizer.optimize_sa_command(base_cmd)
    
    # 최적화 설정 로드
    config = get_sa_config_by_jti(jti_code)
    
    # 기본 명령어에서 시작
    cmd = base_cmd.copy()
    
    # 성능 최적화 먼저 적용
    cmd = optimizer.optimize_sa_command(cmd)
    
    # SA 최적화 설정 적용
    sa_config = config.get('sa_config', {}) if config else {}
    
    # 실제 텍스트명 가져오기
    actual_text_name = get_text_name_by_jti(jti_code)
    
    print(f"🎯 SA 최적화 적용: JTI {jti_code} → {actual_text_name}")
    if config:
        print(f"   성능 등급: {config.get('performance_grade', 'N/A')}, 목표 F1: {config.get('target_f1', 'N/A')}")
    else:
        print(f"   ⚠️ 최적화 설정 없음 - 기본 성능 최적화만 적용")
    
    # SA 파라미터 적용
    sa_param_map = {
        'min_src_tokens': '--min-src-tokens',
        'max_src_tokens': '--max-src-tokens', 
        'min_tgt_tokens': '--min-tgt-tokens',
        'max_tgt_tokens': '--max-tgt-tokens',
        'dp_window': '--dp-window',
        'distance_decay': '--distance-decay',
        'boundary_bonus': '--boundary-bonus',
        'particle_bonus': '--particle-bonus',
        'length_penalty': '--length-penalty',
        'sim_gamma': '--sim-gamma',
        'syntax_hints': '--syntax-hints',
        'comma_bonus': '--comma-bonus',
        'comma_mode': '--comma-mode',
        'syntax_when': '--syntax-when'
    }
    
    applied_params = 0
    for config_key, cmd_arg in sa_param_map.items():
        if config_key in sa_config:
            # 기존 파라미터 제거
            cmd = [x for x in cmd if not x.startswith(cmd_arg)]
            # 새 값 추가
            cmd.extend([cmd_arg, str(sa_config[config_key])])
            applied_params += 1
    
    # max_workers 처리
    if 'max_workers' in sa_config:
        try:
            idx = cmd.index('--max-workers')
            cmd[idx + 1] = str(sa_config['max_workers'])
            applied_params += 1
        except ValueError:
            cmd.extend(['--max-workers', str(sa_config['max_workers'])])
            applied_params += 1
    
    print(f"   SA 최적화 파라미터 {applied_params}개 적용됨")
    
    return cmd

def enable_xml_pipeline_optimization():
    """XML 파이프라인 최적화 활성화 상태 확인"""
    try:
        from config.text_optimization_configs import OPTIMIZATION_CONFIGS
        return len(OPTIMIZATION_CONFIGS) > 0
    except ImportError:
        return False

# 사용 예시 (기존 코드에 통합하는 방법)
"""
기존 코드에서:

cmd = [
    sys.executable, 
    str(pa_main_path), 
    str(input_file), 
    str(pa_output),
    "--embedder", "bge",
    "--max-workers", "2"
]

이렇게 변경:

base_cmd = [
    sys.executable, 
    str(pa_main_path), 
    str(input_file), 
    str(pa_output),
    "--embedder", "bge",
    "--max-workers", "2"
]
cmd = build_optimized_pa_cmd(xml_pair, input_file, pa_output, base_cmd)
"""