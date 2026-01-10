#!/usr/bin/env python3
"""PA 무결성 리포트 (하위 호환성 wrapper)

이 파일은 하위 호환성을 위한 wrapper입니다.
실제 구현은 accuracy/pa_evaluator.py에 있습니다.

Usage:
    python integrity_report.py --input <pa_output> --gold <gold_sentences>
    python accuracy/pa_evaluator.py --input <pa_output> --gold <gold_sentences>
"""

import sys
from pathlib import Path

# accuracy 폴더를 path에 추가
accuracy_dir = Path(__file__).parent / "accuracy"
if str(accuracy_dir) not in sys.path:
    sys.path.insert(0, str(accuracy_dir))

# pa_evaluator의 모든 함수와 클래스를 import하여 재export
from pa_evaluator import *

# main 실행
if __name__ == "__main__":
    import pa_evaluator
    sys.exit(pa_evaluator.main())
