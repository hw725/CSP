#!/bin/bash
# S2P 빠른 테스트 (소규모 샘플, ~100문장)
# Docker 내부에서 실행. 전체 테스트 전 빠른 확인용.
set -e

echo "========================================"
echo "S2P 빠른 테스트 (소규모)"
echo "========================================"

INPUT="datasets/splits/sentence_test.xlsx"
GOLD="datasets/splits/phrase_test.xlsx"
OUTPUT_DIR="datasets/gold"
mkdir -p "$OUTPUT_DIR" test_results

# 소규모 입력 생성 (처음 100행만)
python -c "
import pandas as pd
df = pd.read_excel('$INPUT')
df.head(100).to_excel('${OUTPUT_DIR}/sentence_test_sample.xlsx', index=False)
print(f'샘플 생성: {min(100, len(df))}행')
"
SAMPLE_INPUT="$OUTPUT_DIR/sentence_test_sample.xlsx"

# 개선 버전 테스트
echo ""
echo "🔹 개선 버전 테스트 (100행 샘플)..."
SAMPLE_OUT="$OUTPUT_DIR/s2p_output_sample.csv"

python -m s2p.main \
    "$SAMPLE_INPUT" \
    "$SAMPLE_OUT" \
    --use-boundary-model \
    --boundary-threshold 0.55 \
    --preload-models \
    --syntax-hints ko \
    --similarity-threshold 0.50 \
    --sim-gamma 1.0 \
    --chunk-size 200 \
    --batch-size 64 \
    -v

echo ""
echo "📊 평가..."
python scripts/quick_s2p_eval.py "$GOLD" "$SAMPLE_OUT" \
    -o "test_results/s2p_sample_quick.json"

python accuracy/s2p_evaluator.py "$GOLD" "$SAMPLE_OUT" -v \
    -o "test_results/s2p_sample_eval.csv"

echo ""
echo "✅ 빠른 테스트 완료"
