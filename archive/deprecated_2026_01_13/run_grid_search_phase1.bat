@echo off
REM Grid Search Phase 1 배치 실행 스크립트
REM Prior Bonus (0.10, 0.15, 0.20) × Seeds (1, 2, 3)
REM 총 9회 실험

echo ================================================================================
echo Grid Search Phase 1 배치 실행 시작
echo ================================================================================
echo 시작 시간: %date% %time%
echo.

python scripts/grid_search_pa_weights.py ^
  --prior-bonus 0.10,0.15,0.20 ^
  --seeds 1,2,3 ^
  --output-dir test_results/grid_search_phase1 ^
  --yes

echo.
echo ================================================================================
echo Grid Search Phase 1 배치 실행 완료
echo ================================================================================
echo 종료 시간: %date% %time%
echo.
echo 결과 집계 실행 중...
python scripts/summarize_grid_search.py test_results/grid_search_phase1

pause
