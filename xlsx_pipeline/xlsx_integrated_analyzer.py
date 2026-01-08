#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XLSX 파이프라인 통합 분석기
PA/SA/Accuracy 기능을 Excel 데이터에 통합 적용
"""

import pandas as pd
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import json
from common.progress_manager import (
    start_unified_progress,
    update_unified_progress,
    finish_unified_progress,
    set_progress_description,
)

# 재귀 제한 증가 (PA 무결성 검증 때문에 필요)
sys.setrecursionlimit(10000)

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# PA/SA/Accuracy 모듈 import
try:
    from pa.aligner import process_paragraph_alignment
    PA_AVAILABLE = True
except ImportError as e:
    PA_AVAILABLE = False
    logging.warning(f"⚠️ PA 모듈을 로드할 수 없습니다: {e}")

try:
    from sa.sa_aligner import process_sa_alignment
    SA_AVAILABLE = True
except ImportError:
    SA_AVAILABLE = False
    logging.warning("⚠️ SA 모듈을 로드할 수 없습니다.")

try:
    from accuracy.accuracy_evaluator import AccuracyEvaluator
    ACCURACY_AVAILABLE = True
except ImportError:
    ACCURACY_AVAILABLE = False
    logging.warning("⚠️ Accuracy 모듈을 로드할 수 없습니다.")

logger = logging.getLogger(__name__)


class XLSXIntegratedAnalyzer:
    """Excel 데이터에 대한 통합 분석기"""
    
    def __init__(self, output_dir: str = "xlsx_pipeline_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 모듈 가용성 체크
        self.modules_available = {
            "pa": PA_AVAILABLE,
            "sa": SA_AVAILABLE,
            "accuracy": ACCURACY_AVAILABLE
        }
        
        logger.info(f"📦 모듈 가용성: {self.modules_available}")
    
    def run_pa_analysis(self, paragraph_df: pd.DataFrame, book_id: str, use_global_progress: bool = False) -> Dict[str, Any]:
        """
        문단병렬 데이터에 PA 분석 실행
        
        Args:
            paragraph_df: 문단병렬 데이터프레임 (문단식별자, 원문, 번역문)
            book_id: 책 식별자
        
        Returns:
            PA 분석 결과
        """
        if not PA_AVAILABLE:
            return {"error": "PA 모듈을 사용할 수 없습니다"}
        
        logger.info(f"🔍 PA 분석 시작: {book_id}")
        if not use_global_progress:
            start_unified_progress(total=len(paragraph_df))
            set_progress_description(f"PA 진행: {book_id}")
        embedder_name = os.getenv("CSP_EMBEDDER", "bge").lower()
        device = os.getenv("CSP_DEVICE", "cuda")
        logger.info(f"⚙️ PA 임베더: {embedder_name} (device={device})")
        
        try:
            # 각 문단쌍에 대해 PA 처리
            result_rows = []
            
            for idx, row in paragraph_df.iterrows():
                src_paragraph = str(row.get('원문', ''))
                tgt_paragraph = str(row.get('번역문', ''))
                
                if not src_paragraph.strip() or not tgt_paragraph.strip():
                    continue
                
                # PA 처리: 문단 내 문장 정렬 (BGE 임베더 사용)
                alignments = process_paragraph_alignment(
                    src_paragraph=src_paragraph,
                    tgt_paragraph=tgt_paragraph,
                    embedder_name=embedder_name,  # 기본 BGE, 환경변수로 openai 등 전환 가능
                    tokenizer_name='siku',
                    max_length=150,
                    similarity_threshold=0.3,
                    device=device,  # GPU 기본, 필요 시 CPU/None
                    use_spacy_tokenizer=False,  # spaCy 비활성화로 재귀 방지
                    max_workers=4,
                    batch_size=100
                )
                
                # 결과를 행으로 변환
                for alignment in alignments:
                    result_row = row.to_dict()
                    result_row['원문'] = alignment.get('원문', '')
                    result_row['번역문'] = alignment.get('번역문', '')
                    result_row['similarity'] = alignment.get('similarity', 0.0)
                    result_rows.append(result_row)

                # 진행률 업데이트
                update_unified_progress(advance=1)
            
            # 결과 DataFrame 생성
            result_df = pd.DataFrame(result_rows)
            
            # 결과 저장
            output_file = self.output_dir / f"{book_id}_pa_output.xlsx"
            result_df.to_excel(output_file, index=False)
            
            logger.info(f"✅ PA 분석 완료: {output_file}")
            if not use_global_progress:
                finish_unified_progress()
            
            return {
                "success": True,
                "output_file": str(output_file),
                "input_rows": len(paragraph_df),
                "output_rows": len(result_df),
                "columns": list(result_df.columns)
            }
            
        except Exception as e:
            logger.error(f"❌ PA 분석 실패: {e}")
            import traceback
            traceback.print_exc()
            if not use_global_progress:
                finish_unified_progress()
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_sa_analysis(self, sentence_df: pd.DataFrame, book_id: str, use_global_progress: bool = False) -> Dict[str, Any]:
        """
        문장병렬 데이터에 SA 분석 실행
        
        Args:
            sentence_df: 문장병렬 데이터프레임 (문단식별자, 문장식별자, 원문, 번역문)
            book_id: 책 식별자
        
        Returns:
            SA 분석 결과
        """
        if not SA_AVAILABLE:
            return {"error": "SA 모듈을 사용할 수 없습니다"}
        
        logger.info(f"🔍 SA 분석 시작: {book_id}")
        if not use_global_progress:
            start_unified_progress(total=len(sentence_df))
            set_progress_description(f"SA 진행: {book_id}")
        sa_embedder = os.getenv("CSP_SA_EMBEDDER", "bge").lower()
        sa_use_semantic = os.getenv("CSP_SA_USE_SEMANTIC", "1") != "0"
        sa_max_workers = int(os.getenv("CSP_SA_WORKERS", "4"))
        sa_batch_size = int(os.getenv("CSP_SA_BATCH", "100"))
        sa_device = os.getenv("CSP_DEVICE", "cuda")
        sa_device_id = 0 if sa_device.lower() == "cuda" else None
        logger.info(f"⚙️ SA 임베더: {sa_embedder}, device={sa_device}, semantic={sa_use_semantic}, workers={sa_max_workers}, batch={sa_batch_size}")
        
        try:
            # 각 문장쌍에 대해 SA 처리
            result_rows = []
            
            for idx, row in sentence_df.iterrows():
                src_text = str(row.get('원문', ''))
                translation = str(row.get('번역문', ''))
                
                if not src_text.strip() or not translation.strip():
                    continue

                if idx % 200 == 0:
                    logger.info(f"... SA 진행 {idx}/{len(sentence_df)}")
                
                # SA 처리: 원문과 번역문을 단위별로 정렬
                aligned_units = process_sa_alignment(
                    src_text,
                    translation,
                    embedder=sa_embedder,
                    embedder_name=sa_embedder,
                    embedder_device=sa_device,
                    embedder_device_id=sa_device_id,
                    use_semantic=sa_use_semantic,
                    max_workers=sa_max_workers,
                    batch_size=sa_batch_size,
                )
                
                # 결과를 행으로 변환
                src_units = aligned_units.get('source_units', [])
                tgt_units = aligned_units.get('translation_units', [])
                similarities = aligned_units.get('similarities', [])

                # 🔧 안전장치: 비어 있거나 길이가 다르면 원문/번역을 패딩하여 최소 1행 보장
                if not src_units and not tgt_units:
                    src_units = [src_text]
                    tgt_units = [translation]
                    similarities = [0.0]
                max_len = max(len(src_units), len(tgt_units), 1)
                src_units = (src_units + [''] * max_len)[:max_len]
                tgt_units = (tgt_units + [''] * max_len)[:max_len]
                similarities = (similarities + [0.0] * max_len)[:max_len]
                
                for src_unit, tgt_unit, similarity in zip(src_units, tgt_units, similarities):
                    result_row = row.to_dict()
                    result_row['원문'] = src_unit
                    result_row['번역문'] = tgt_unit
                    result_row['유사도'] = round(similarity, 4)
                    result_rows.append(result_row)

                # 진행률 업데이트 (문장 단위 증가)
                update_unified_progress(advance=1)
            
            # 결과 DataFrame 생성
            result_df = pd.DataFrame(result_rows)
            
            # 결과 저장
            output_file = self.output_dir / f"{book_id}_sa_output.xlsx"
            result_df.to_excel(output_file, index=False)
            
            logger.info(f"✅ SA 분석 완료: {output_file}")
            if not use_global_progress:
                finish_unified_progress()
            
            return {
                "success": True,
                "output_file": str(output_file),
                "input_rows": len(sentence_df),
                "output_rows": len(result_df),
                "columns": list(result_df.columns)
            }
            
        except Exception as e:
            logger.error(f"❌ SA 분석 실패: {e}")
            import traceback
            traceback.print_exc()
            if not use_global_progress:
                finish_unified_progress()
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_accuracy_evaluation(
        self,
        ground_truth_file: str,
        prediction_file: str,
        book_id: str,
        project: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        정확도 평가 실행
        
        Args:
            ground_truth_file: 정답 파일 (구병렬)
            prediction_file: 예측 파일 (PA/SA 출력)
            book_id: 책 식별자
            project: 프로젝트 이름 (임계값 설정용)
        
        Returns:
            정확도 평가 결과
        """
        if not ACCURACY_AVAILABLE:
            return {"error": "Accuracy 모듈을 사용할 수 없습니다"}
        
        logger.info(f"📊 정확도 평가 시작: {book_id}")
        
        try:
            # AccuracyEvaluator 초기화
            evaluator = AccuracyEvaluator(
                ground_truth_file=ground_truth_file,
                prediction_file=prediction_file,
                project=project
            )
            
            # 데이터 로드 및 전처리
            evaluator.load_and_prepare_data()
            
            # 정확도 계산
            accuracy_results = evaluator.evaluate_accuracy()
            
            # 결과 저장
            output_file = self.output_dir / f"{book_id}_accuracy.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(accuracy_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ 정확도 평가 완료: {output_file}")
            
            return {
                "success": True,
                "output_file": str(output_file),
                "accuracy": accuracy_results.get("overall_accuracy", 0),
                "total_groups": accuracy_results.get("total_groups", 0),
                "matched_groups": accuracy_results.get("matched_groups", 0)
            }
            
        except Exception as e:
            logger.error(f"❌ 정확도 평가 실패: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_full_pipeline(
        self,
        word_df: pd.DataFrame,
        sentence_df: pd.DataFrame,
        paragraph_df: pd.DataFrame,
        book_id: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        전체 파이프라인 실행
        
        Args:
            word_df: 구병렬 데이터
            sentence_df: 문장병렬 데이터
            paragraph_df: 문단병렬 데이터
            book_id: 책 식별자
            config: 파이프라인 설정
                - run_pa: bool (PA 실행 여부)
                - run_sa: bool (SA 실행 여부)
                - run_accuracy: bool (정확도 평가 여부)
                - project: str (프로젝트 이름)
        
        Returns:
            전체 파이프라인 실행 결과
        """
        config = config or {}
        run_pa = config.get("run_pa", True)
        run_sa = config.get("run_sa", True)
        run_accuracy = config.get("run_accuracy", False)
        project = config.get("project")
        
        results = {
            "book_id": book_id,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "results": {}
        }
        
        # PA 분석
        if run_pa:
            results["results"]["pa"] = self.run_pa_analysis(paragraph_df, book_id)
        
        # SA 분석
        if run_sa:
            results["results"]["sa"] = self.run_sa_analysis(sentence_df, book_id)
        
        # 정확도 평가
        if run_accuracy:
            # 구병렬을 정답으로, PA/SA 출력을 예측으로 사용
            word_file = self.output_dir / f"temp_word_{book_id}.xlsx"
            word_df.to_excel(word_file, index=False)
            
            if run_pa and results["results"]["pa"].get("success"):
                pa_output = results["results"]["pa"]["output_file"]
                results["results"]["pa_accuracy"] = self.run_accuracy_evaluation(
                    str(word_file), pa_output, f"{book_id}_pa", project
                )
            
            if run_sa and results["results"]["sa"].get("success"):
                sa_output = results["results"]["sa"]["output_file"]
                results["results"]["sa_accuracy"] = self.run_accuracy_evaluation(
                    str(word_file), sa_output, f"{book_id}_sa", project
                )
            
            # 임시 파일 삭제
            if word_file.exists():
                word_file.unlink()
        
        # 최종 결과 저장
        result_file = self.output_dir / f"{book_id}_full_pipeline.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 전체 파이프라인 완료: {result_file}")
        
        return results


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 테스트
    analyzer = XLSXIntegratedAnalyzer()
    print(f"📦 모듈 가용성: {analyzer.modules_available}")
