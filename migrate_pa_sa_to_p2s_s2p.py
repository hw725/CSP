#!/usr/bin/env python3
"""
PA/SA → P2S/S2P 마이그레이션 스크립트

명명 규칙:
- 코드 폴더: pa → p2s, sa → s2p
- 데이터셋 폴더: pd → paragraph, pa → sentence, sa → phrase
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent

# 제외할 경로 패턴 (정규식)
EXCLUDE_PATTERNS = [
    r'\.git[\\/]',
    r'\.venv[\\/]',
    r'\.history[\\/]',
    r'__pycache__[\\/]',
    r'\.pyc$',
    r'node_modules[\\/]',
    r'cache[\\/]',
    r'\.pkl$',
    r'archive[\\/]',
    r'hardneg_w1\.5',  # junction 폴더들 제외
    r'migrate_pa_sa_to_p2s_s2p\.py$',  # 이 스크립트 자체 제외!
]

# 1단계: 폴더 이름 변경 (순서 중요: 깊은 것부터)
FOLDER_RENAMES = [
    # 핵심 코드 폴더
    ('pa', 'p2s'),
    ('sa', 's2p'),
    # datasets 폴더
    ('datasets/pd_boundary', 'datasets/paragraph_boundary'),
    ('datasets/pd', 'datasets/paragraph'),
    ('datasets/pa_src_boundary_hardneg', 'datasets/sentence_src_boundary_hardneg'),
    ('datasets/pa_src_boundary', 'datasets/sentence_src_boundary'),
    ('datasets/pa_particle_role', 'datasets/sentence_particle_role'),
    ('datasets/pa_boundary', 'datasets/sentence_boundary'),
    ('datasets/pa', 'datasets/sentence'),
    ('datasets/sa_boundary', 'datasets/phrase_boundary'),
    ('datasets/sa', 'datasets/phrase'),
    # hyeonto/reports 폴더
    ('hyeonto/reports/pa_boundary_k4_full', 'hyeonto/reports/sentence_boundary_k4_full'),
    ('hyeonto/reports/pa_boundary_k14_full', 'hyeonto/reports/sentence_boundary_k14_full'),
    ('hyeonto/reports/sa_boundary_k4_full', 'hyeonto/reports/phrase_boundary_k4_full'),
    ('hyeonto/reports/sa_boundary_k24_full', 'hyeonto/reports/phrase_boundary_k24_full'),
    ('hyeonto/reports/exploratory/cluster_flow_pa', 'hyeonto/reports/exploratory/cluster_flow_sentence'),
    ('hyeonto/reports/exploratory/cluster_flow_sa', 'hyeonto/reports/exploratory/cluster_flow_phrase'),
]

# 2단계: 파일 이름 변경 패턴 (정규식)
FILE_RENAME_PATTERNS = [
    # accuracy 폴더
    (r'^pa_evaluator\.py$', 'p2s_evaluator.py'),
    (r'^sa_evaluator\.py$', 's2p_evaluator.py'),
    # docs 폴더
    (r'^PA_CODE_ANATOMY\.md$', 'P2S_CODE_ANATOMY.md'),
    (r'^PA_MECHANISM\.md$', 'P2S_MECHANISM.md'),
    (r'^SA_CODE_ANATOMY\.md$', 'S2P_CODE_ANATOMY.md'),
    (r'^SA_MECHANISM\.md$', 'S2P_MECHANISM.md'),
    # s2p 폴더 내 파일 (sa_aligner.py 등)
    (r'^sa_aligner\.py$', 's2p_aligner.py'),
    # common 폴더
    (r'^sa_boundary_tagger_loader\.py$', 's2p_boundary_tagger_loader.py'),
    (r'^sa_crossattn_boundary_loader\.py$', 's2p_crossattn_boundary_loader.py'),
    (r'^sa_semantic_boundary_loader\.py$', 's2p_semantic_boundary_loader.py'),
    # scripts 폴더
    (r'^pa_multitest_runner\.py$', 'p2s_multitest_runner.py'),
]

# 3단계: 파일 내용 치환 패턴 (순서 중요: 긴 것부터)
CONTENT_REPLACEMENTS = [
    # import 문 및 모듈 참조
    ('from sa.', 'from s2p.'),
    ('from pa.', 'from p2s.'),
    ('import sa.', 'import s2p.'),
    ('import pa.', 'import p2s.'),
    ('sa/', 's2p/'),
    ('pa/', 'p2s/'),
    # 데이터셋 경로
    ('datasets/pd_boundary', 'datasets/paragraph_boundary'),
    ('datasets/pd', 'datasets/paragraph'),
    ('datasets/pa_src_boundary', 'datasets/sentence_src_boundary'),
    ('datasets/pa_particle_role', 'datasets/sentence_particle_role'),
    ('datasets/pa_boundary', 'datasets/sentence_boundary'),
    ('datasets/pa', 'datasets/sentence'),
    ('datasets/sa_boundary', 'datasets/phrase_boundary'),
    ('datasets/sa', 'datasets/phrase'),
    # hyeonto 리포트 경로
    ('hyeonto/reports/pa_boundary', 'hyeonto/reports/sentence_boundary'),
    ('hyeonto/reports/sa_boundary', 'hyeonto/reports/phrase_boundary'),
    ('cluster_flow_pa', 'cluster_flow_sentence'),
    ('cluster_flow_sa', 'cluster_flow_phrase'),
    # evaluator 참조
    ('from accuracy.pa_evaluator', 'from accuracy.p2s_evaluator'),
    ('from accuracy.sa_evaluator', 'from accuracy.s2p_evaluator'),
    ('pa_evaluator', 'p2s_evaluator'),
    ('sa_evaluator', 's2p_evaluator'),
    # common 모듈 참조
    ('from common.sa_boundary', 'from common.s2p_boundary'),
    ('from common.sa_crossattn', 'from common.s2p_crossattn'),
    ('from common.sa_semantic', 'from common.s2p_semantic'),
    ('sa_aligner', 's2p_aligner'),
    # 클러스터 프로파일
    ('pa_cluster_profile', 'sentence_cluster_profile'),
    ('sa_cluster_profile', 'phrase_cluster_profile'),
]

def should_exclude(path: Path) -> bool:
    """제외 대상인지 확인"""
    path_str = str(path)
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, path_str):
            return True
    try:
        if path.is_symlink():
            return True
    except OSError:
        return True
    return False

def run_migration(dry_run: bool = True):
    """마이그레이션 실행"""
    print(f"{'[DRY RUN] ' if dry_run else ''}PA/SA → P2S/S2P 마이그레이션")
    print("=" * 60)
    
    # 1. 폴더 이름 변경
    print("\n## 1. 폴더 이름 변경")
    for old_rel, new_rel in FOLDER_RENAMES:
        old_path = PROJECT_ROOT / old_rel
        new_path = PROJECT_ROOT / new_rel
        if old_path.exists() and old_path.is_dir():
            print(f"  {old_rel} → {new_rel}")
            if not dry_run:
                try:
                    if new_path.exists():
                        print(f"    [SKIP] 대상이 이미 존재함")
                    else:
                        new_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(old_path), str(new_path))
                except Exception as e:
                    print(f"    [ERROR] {e}")
    
    # 2. 파일 이름 변경
    print("\n## 2. 파일 이름 변경")
    file_renames = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        root_path = Path(root)
        if should_exclude(root_path):
            dirs[:] = []
            continue
        
        for filename in files:
            file_path = root_path / filename
            if should_exclude(file_path):
                continue
            
            for pattern, replacement in FILE_RENAME_PATTERNS:
                if re.match(pattern, filename):
                    new_name = re.sub(pattern, replacement, filename)
                    new_path = root_path / new_name
                    if new_name != filename:
                        file_renames.append((file_path, new_path))
                    break
    
    for old_path, new_path in file_renames:
        print(f"  {old_path.relative_to(PROJECT_ROOT)} → {new_path.name}")
        if not dry_run:
            try:
                if new_path.exists():
                    print(f"    [SKIP] 대상이 이미 존재함")
                else:
                    shutil.move(str(old_path), str(new_path))
            except Exception as e:
                print(f"    [ERROR] {e}")
    
    # 3. 파일 내용 업데이트
    print("\n## 3. 파일 내용 업데이트")
    extensions = {'.py', '.md', '.json', '.txt', '.yml', '.yaml'}
    updated_count = 0
    
    for root, dirs, files in os.walk(PROJECT_ROOT):
        root_path = Path(root)
        if should_exclude(root_path):
            dirs[:] = []
            continue
        
        for filename in files:
            file_path = root_path / filename
            if file_path.suffix.lower() not in extensions:
                continue
            if should_exclude(file_path):
                continue
            
            try:
                content = file_path.read_text(encoding='utf-8')
            except:
                try:
                    content = file_path.read_text(encoding='cp949')
                except:
                    continue
            
            original = content
            for old, new in CONTENT_REPLACEMENTS:
                content = content.replace(old, new)
            
            if content != original:
                print(f"  [UPDATE] {file_path.relative_to(PROJECT_ROOT)}")
                updated_count += 1
                if not dry_run:
                    try:
                        file_path.write_text(content, encoding='utf-8')
                    except Exception as e:
                        print(f"    [ERROR] {e}")
    
    print(f"\n  총 {updated_count}개 파일 업데이트 {'예정' if dry_run else '완료'}")
    
    print("\n" + "=" * 60)
    if dry_run:
        print("위는 DRY RUN 결과입니다. 실제 실행하려면:")
        print("  python migrate_pa_sa_to_p2s_s2p.py --execute")
    else:
        print("마이그레이션 완료!")

if __name__ == "__main__":
    import sys
    dry_run = "--execute" not in sys.argv
    run_migration(dry_run=dry_run)
