#!/usr/bin/env python3
"""
현토 분석 전체 시각화 통합 생성 스크립트

모든 시각화를 한 번에 생성합니다:
1. 임베딩 오버레이 (2D/3D)
2. Sankey 다이어그램
3. 흑백 인쇄용 시각화 (generate_visualizations_bw.py)

--grayscale 옵션: 흑백 버전도 함께 생성
--only-grayscale: 흑백 버전만 생성
"""

import subprocess
import sys
from pathlib import Path
import argparse
from datetime import datetime


BASE_DIR = Path(__file__).parent


def run_script(script_name: str, args: list = None):
    """스크립트 실행"""
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        print(f"⚠️  스크립트 없음: {script_name}")
        return False
    
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    print(f"\n{'='*60}")
    print(f"▶ {script_name} {' '.join(args or [])}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='모든 시각화 생성')
    parser.add_argument('--grayscale', action='store_true', 
                        help='컬러 + 흑백 버전 모두 생성')
    parser.add_argument('--only-grayscale', action='store_true', 
                        help='흑백 버전만 생성')
    parser.add_argument('--skip-umap', action='store_true',
                        help='UMAP 임베딩 건너뛰기 (시간 절약)')
    args = parser.parse_args()
    
    print("="*70)
    print("🎨 현토 분석 전체 시각화 생성")
    print("="*70)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    mode = "only-grayscale" if args.only_grayscale else ("both" if args.grayscale else "color")
    print(f"모드: {mode}")
    
    scripts_executed = 0
    scripts_failed = 0
    
    # 1. 컬러 버전 생성 (only-grayscale이 아닌 경우)
    if not args.only_grayscale:
        print("\n" + "="*70)
        print("📌 STEP 1: 컬러 버전 시각화 생성")
        print("="*70)
        
        # 1-1. 임베딩 오버레이 (2D/3D)
        if not args.skip_umap:
            if run_script("analyze_embedding_overlay.py", ["--sample", "10000"]):
                scripts_executed += 1
            else:
                scripts_failed += 1
        
        # 1-2. Sankey 다이어그램
        if run_script("generate_sankey_diagrams.py"):
            scripts_executed += 1
        else:
            scripts_failed += 1
    
    # 2. 흑백 버전 생성 (grayscale 또는 only-grayscale인 경우)
    if args.grayscale or args.only_grayscale:
        print("\n" + "="*70)
        print("📌 STEP 2: 흑백 인쇄용 시각화 생성")
        print("="*70)
        
        # 2-1. 임베딩 오버레이 (2D/3D) - 흑백
        if not args.skip_umap:
            if run_script("analyze_embedding_overlay.py", ["--sample", "10000", "--grayscale"]):
                scripts_executed += 1
            else:
                scripts_failed += 1
        
        # 2-2. Sankey 다이어그램 - 흑백
        if run_script("generate_sankey_diagrams.py", ["--grayscale"]):
            scripts_executed += 1
        else:
            scripts_failed += 1
        
        # 2-3. 전용 흑백 시각화 스크립트
        if run_script("generate_visualizations_bw.py"):
            scripts_executed += 1
        else:
            scripts_failed += 1
    
    # 최종 보고
    print("\n" + "="*70)
    print("✅ 시각화 생성 완료!")
    print("="*70)
    print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"실행된 스크립트: {scripts_executed}개")
    if scripts_failed > 0:
        print(f"⚠️  실패한 스크립트: {scripts_failed}개")
    
    print("\n📁 생성된 시각화 위치:")
    print(f"   - 임베딩 오버레이: reports/k4_embedding_overlay_*.html")
    print(f"   - Sankey 다이어그램: reports/sankey_diagrams/")
    if args.grayscale or args.only_grayscale:
        print(f"   - 흑백 시각화: reports/print_friendly/")


if __name__ == "__main__":
    main()
