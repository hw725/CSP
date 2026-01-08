@echo off
REM XML 파이프라인 실행 스크립트 (Windows)
REM XML 쌍(원문+번역문) 전체 파이프라인 자동 처리

echo ========================================
echo CSP XML 전체 파이프라인 시스템
echo ========================================

REM 인자 확인
if "%1"=="" (
    echo.
    echo 사용법:
    echo   xml_pipeline.bat [명령어] [옵션들]
    echo.
    echo 명령어:
    echo   process  - 단일 XML 쌍 처리
    echo   batch    - 디렉토리 일괄 처리
    echo   list     - 최근 처리 결과 목록
    echo   show     - 특정 쌍 상세 결과
    echo   cleanup  - 오래된 결과 정리
    echo.
    echo 예시:
    echo   # 단일 쌍 처리
    echo   xml_pipeline.bat process 원문.xml 번역문.xml
    echo.
    echo   # 디렉토리 일괄 처리
    echo   xml_pipeline.bat batch "C:\xml_files"
    echo.
    echo   # 최근 결과 목록
    echo   xml_pipeline.bat list
    echo.
    echo   # 상세 결과 조회
    echo   xml_pipeline.bat show 쌍ID
    echo.
    goto :eof
)

REM Python 가상환경 활성화 (Docker가 아닌 경우)
if exist "venv\Scripts\activate.bat" (
    echo Python 가상환경 활성화...
    call venv\Scripts\activate.bat
)

REM 명령어별 처리
if "%1"=="process" goto :process_single
if "%1"=="batch" goto :batch_process
if "%1"=="list" goto :list_results
if "%1"=="show" goto :show_details
if "%1"=="cleanup" goto :cleanup_old

echo 알 수 없는 명령어: %1
goto :eof

:process_single
echo.
echo === 단일 XML 쌍 처리 ===
set ORIGINAL_XML=%2
set TRANSLATION_XML=%3
set PAIR_ID=%4

if "%ORIGINAL_XML%"=="" (
    echo 오류: 원문 XML 파일이 필요합니다.
    echo 사용법: xml_pipeline.bat process [원문.xml] [번역문.xml] [쌍ID]
    goto :eof
)

if "%TRANSLATION_XML%"=="" (
    echo 오류: 번역문 XML 파일이 필요합니다.
    echo 사용법: xml_pipeline.bat process [원문.xml] [번역문.xml] [쌍ID]
    goto :eof
)

echo 원문 XML: %ORIGINAL_XML%
echo 번역문 XML: %TRANSLATION_XML%
if not "%PAIR_ID%"=="" echo 쌍 ID: %PAIR_ID%
echo.

if not "%PAIR_ID%"=="" (
    python xml_pipeline_cli.py process --original "%ORIGINAL_XML%" --translation "%TRANSLATION_XML%" --pair-id "%PAIR_ID%"
) else (
    python xml_pipeline_cli.py process --original "%ORIGINAL_XML%" --translation "%TRANSLATION_XML%"
)

goto :eof

:batch_process
echo.
echo === 디렉토리 일괄 처리 ===
set XML_DIR=%2
set PATTERN=%3

if "%XML_DIR%"=="" (
    echo 오류: XML 디렉토리가 필요합니다.
    echo 사용법: xml_pipeline.bat batch [XML_디렉토리] [패턴]
    goto :eof
)

echo XML 디렉토리: %XML_DIR%
if not "%PATTERN%"=="" (
    echo 파일 패턴: %PATTERN%
    python xml_pipeline_cli.py batch --xml-dir "%XML_DIR%" --pattern "%PATTERN%"
) else (
    echo 파일 패턴: *원문*.xml (기본값)
    python xml_pipeline_cli.py batch --xml-dir "%XML_DIR%"
)

goto :eof

:list_results
echo.
echo === 최근 처리 결과 목록 ===
set LIMIT=%2

if not "%LIMIT%"=="" (
    echo 조회 개수: %LIMIT%
    python xml_pipeline_cli.py list --limit %LIMIT%
) else (
    echo 조회 개수: 10 (기본값)
    python xml_pipeline_cli.py list
)

goto :eof

:show_details
echo.
echo === 상세 결과 조회 ===
set PAIR_ID=%2

if "%PAIR_ID%"=="" (
    echo 오류: XML 쌍 ID가 필요합니다.
    echo 사용법: xml_pipeline.bat show [쌍ID]
    goto :eof
)

echo 쌍 ID: %PAIR_ID%
echo.

python xml_pipeline_cli.py show "%PAIR_ID%"

goto :eof

:cleanup_old
echo.
echo === 오래된 결과 정리 ===
set DAYS=%2

if not "%DAYS%"=="" (
    echo 보관 기간: %DAYS%일
    python xml_pipeline_cli.py cleanup --days %DAYS%
) else (
    echo 보관 기간: 7일 (기본값)
    python xml_pipeline_cli.py cleanup
)

goto :eof