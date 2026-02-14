"""P2S (Paragraph to Sentence) package — splits paragraph-level parallel text into sentence-level aligned pairs.

P2S (문단→문장 분할) 패키지

핵심 기능:
- 한문 원문-번역문 문단을 문장 단위로 분할
- 다중 전략 후보 생성 (SuPar, 경계 모델, DP, TopK)
- BGE refinement을 통한 경계 최적화
- 100% 텍스트 무결성 보장
"""

__version__ = "3.0.0"
