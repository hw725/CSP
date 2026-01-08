"""
프로젝트별(PA/SA) 임계값 설정.
퍼센타일 기반 제안치(P50/P75/P90)를 최소/권장/상위로 매핑.

단위: 기본 row(행 단위) 기준.
핵심 지표: partial_match, target_avg_similarity
"""

THRESHOLDS = {
    # Paragraph Alignment (PA): row 평가 기준
    'pa': {
        'unit': 'row',
        'metrics': ['partial_match', 'target_avg_similarity'],
        'levels': {
            # ≈ P50
            'min': {
                'partial_match': 0.10,
                'target_avg_similarity': 0.10,
            },
            # ≈ P75
            'recommended': {
                'partial_match': 0.15,
                'target_avg_similarity': 0.19,
            },
            # ≈ P90
            'top': {
                'partial_match': 0.21,
                'target_avg_similarity': 0.26,
            },
        },
    },

    # Sentence Alignment (SA): row 평가 기준
    'sa': {
        'unit': 'row',
        'metrics': ['partial_match', 'target_avg_similarity'],
        'levels': {
            # ≈ P50
            'min': {
                'partial_match': 0.885,
                'target_avg_similarity': 0.769,
            },
            # ≈ P75
            'recommended': {
                'partial_match': 0.952,
                'target_avg_similarity': 0.905,
            },
            # ≈ P90
            'top': {
                'partial_match': 1.0,
                'target_avg_similarity': 1.0,
            },
        },
    },
}

LABEL_ORDER = ['below', 'min', 'recommended', 'top']
