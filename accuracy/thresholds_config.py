"""
프로젝트별(PA/SA) 임계값 설정.
이제 csp_config.json에서 중앙 관리됩니다.
하위 호환성을 위해 common.config에서 로드합니다.
"""

from common.config import get_thresholds

# 하위 호환성을 위한 직접 import
THRESHOLDS = get_thresholds()

LABEL_ORDER = ["below", "min", "recommended", "top"]
