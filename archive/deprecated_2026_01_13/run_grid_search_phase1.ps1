# Grid Search Phase 1 배치 실행 스크립트 (PowerShell)
# Prior Bonus (0.10, 0.15, 0.20) × Seeds (1, 2, 3)
# 총 9회 실험

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Grid Search Phase 1 배치 실행 시작" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "시작 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host ""

# 로그 파일 설정
$logFile = "test_results/grid_search_phase1_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# Grid Search 실행
python scripts/grid_search_pa_weights.py `
  --prior-bonus 0.10,0.15,0.20 `
  --seeds 1,2,3 `
  --output-dir test_results/grid_search_phase1 `
  --yes | Tee-Object -FilePath $logFile

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Grid Search Phase 1 배치 실행 완료" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "종료 시간: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host "로그 파일: $logFile" -ForegroundColor Yellow
Write-Host ""

# 결과 집계
Write-Host "결과 집계 실행 중..." -ForegroundColor Yellow
python scripts/summarize_grid_search.py test_results/grid_search_phase1

Write-Host ""
Write-Host "완료! 결과를 확인하세요." -ForegroundColor Green
