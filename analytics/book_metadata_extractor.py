import re
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class BookMetadataExtractor:
    def __init__(self):
        self.basic_text_mappings = {
            '논어': '공자', '맹자': '맹자', '관자': '관중', '안씨가훈': '안지추'
        }
        self.default_translator = '한국고전번역원'
        
        # 작가별 시대 정보 (추론용)
        self.author_period_map = {
            '공자': '춘추시대(BC 551-479)',
            '맹자': '전국시대(BC 372-289)', 
            '관중': '춘추시대(BC ?-645)',
            '안지추': '북제시대(531-591)',
            '양웅': '전한시대(BC 53-AD 18)',
            '한유': '당대(768-824)',
            '유종원': '당대(773-819)',
            '구양수': '북송시대(1007-1072)',
            '소식': '북송시대(1037-1101)',
            '소철': '북송시대(1009-1066)',
            '소순': '북송시대(1039-1112)',
            '증공': '북송시대(1019-1083)',
            '왕안석': '북송시대(1021-1086)',
            '주희': '남송시대(1130-1200)',
            '진덕수': '남송시대(1178-1235)',
            '육지': '당대(754-805)',  # 수정: 삼국위시대 -> 당대
            '송기채': '조선시대(역자)',
            '정태현': '조선시대(역자)',
            '이상하': '조선시대(역자)',
            '성백효': '조선시대(역자)',
            '김동주': '조선시대(역자)',
            '신용호': '조선시대(역자)',
            '허호구': '조선시대(역자)',
        }
    
    def extract_jti_code_from_filename(self, filename: str) -> Optional[str]:
        jti_match = re.search(r'jti_([0-9a-z]+)', filename.lower())
        if jti_match:
            return jti_match.group(1)
        return None
    
    def get_sibu_classification(self, book_name: str) -> str:
        jti_code = self.extract_jti_code_from_filename(book_name)
        if jti_code and len(jti_code) > 0:
            sibu_map = {'1': '經', '2': '史', '3': '子', '4': '集'}
            return sibu_map.get(jti_code[0], '未詳')
        return '子' if '관자' in book_name else '未詳'
    
    def extract_text_name_from_filename(self, filename: str) -> Optional[str]:
        bracket_match = re.search(r'\[.*?\]([^_\-0-9]+)', filename)
        if bracket_match:
            text_name = bracket_match.group(1).strip()
            text_name = re.sub(r'\d+$', '', text_name).strip()
            return text_name if text_name else None
        
        simple_match = re.search(r'^([가-힣]+)', filename)
        return simple_match.group(1) if simple_match else None
    
    def get_basic_author_from_text_name(self, text_name: str) -> str:
        for name, author in self.basic_text_mappings.items():
            if name in text_name:
                return author
        
        if '당송팔대가문초' in text_name:
            authors = ['한유', '유종원', '구양수', '소식', '소철', '소순', '증공', '왕안석']
            for author in authors:
                if author in text_name:
                    return author
        
        return '미상'
    
    def get_period_from_author(self, author_name: str) -> str:
        """작가명으로부터 시대 정보 추론"""
        if not author_name or author_name == '미상':
            return '미상'
        return self.author_period_map.get(author_name, '미상')
    
    def extract_metadata(self, book_name: str) -> Dict[str, str]:
        author, translator = self.extract_author_translator(book_name)
        period = self.get_period_from_author(author)
        return {
            'author': author, 
            'translator': translator,
            'period': period,
            'sibu_classification': self.get_sibu_classification(book_name)
        }
    
    def extract_author_translator(self, book_name: str) -> Tuple[str, str]:
        try:
            text_name = self.extract_text_name_from_filename(book_name)
            if text_name:
                author = self.get_basic_author_from_text_name(text_name)
                return author, self.default_translator
            return '미상', self.default_translator
        except Exception as e:
            logger.error(f'작가/역자 정보 추출 중 오류: {e}')
            return '미상', self.default_translator
    
    def get_detailed_author_info(self, book_name: str) -> Dict[str, str]:
        author, translator = self.extract_author_translator(book_name)
        return {
            'author': author,
            'translator': translator,
            'period': self.get_period_from_author(author),
            'sibu_classification': self.get_sibu_classification(book_name),
        }