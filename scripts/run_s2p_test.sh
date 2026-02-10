#!/bin/bash
# S2P 기준선 + 개선 테스트 (Docker 내부에서 실행)
# 사용법: bash scripts/run_s2p_test.sh
set -e

echo "========================================"
echo "S2P 테스트 시작"
echo "========================================"

INPUT="datasets/splits/sentence_test.xlsx"
GOLD="datasets/splits/phrase_test.xlsx"
OUTPUT_DIR="datasets/gold"
mkdir -p "$OUTPUT_DIR"

# ===== 1. 기준선 (baseline) 테스트 =====
echo ""
echo "🔹 [1/2] 기준선 테스트 (boundary model 없이)"
echo "========================================"
BASELINE_OUT="$OUTPUT_DIR/s2p_output_baseline.csv"

python -m s2p.main \
    "$INPUT" \
    "$BASELINE_OUT" \
    --no-boundary-model \
    --preload-models \
    --syntax-hints ko \
    --chunk-size 200 \
    --batch-size 64

echo ""
echo "📊 기준선 평가..."
python accuracy/s2p_evaluator.py "$GOLD" "$BASELINE_OUT" -v \
    -o "test_results/s2p_baseline_eval.csv"

python scripts/quick_s2p_eval.py "$GOLD" "$BASELINE_OUT" \
    -o "test_results/s2p_baseline_quick.json"

# ===== 2. 개선 버전 테스트 (boundary model + guard) =====
echo ""
echo "🔹 [2/2] 개선 버전 테스트 (boundary model + refinement guard)"
echo "========================================"
IMPROVED_OUT="$OUTPUT_DIR/s2p_output_improved.csv"

python -m s2p.main \
    "$INPUT" \
    "$IMPROVED_OUT" \
    --use-boundary-model \
    --boundary-threshold 0.55 \
    --preload-models \
    --syntax-hints ko \
    --similarity-threshold 0.50 \
    --sim-gamma 1.0 \
    --chunk-size 200 \
    --batch-size 64

echo ""
echo "📊 개선 버전 평가..."
python accuracy/s2p_evaluator.py "$GOLD" "$IMPROVED_OUT" -v \
    -o "test_results/s2p_improved_eval.csv"

python scripts/quick_s2p_eval.py "$GOLD" "$IMPROVED_OUT" \
    -o "test_results/s2p_improved_quick.json"

# ===== 결과 비교 =====
echo ""
echo "========================================"
echo "📊 결과 비교"
echo "========================================"
python -c "
import json, os
for label, path in [('기준선', 'test_results/s2p_baseline_quick.json'),
                     ('개선', 'test_results/s2p_improved_quick.json')]:
    if os.path.exists(path):
        with open(path) as f:
            r = json.load(f)
        print(f'{label}: F1={r[\"f1\"]:.4f}  P={r[\"precision\"]:.4f}  R={r[\"recall\"]:.4f}  AvgSim={r[\"tgt_avg_sim\"]:.4f}')
    else:
        print(f'{label}: 결과 없음')
"

echo ""
echo "✅ S2P 테스트 완료"
