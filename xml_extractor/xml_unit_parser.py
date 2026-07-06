#!/usr/bin/env python3
"""
XML 단위별 파서
XML 파일에서 문장(<s>) 단위와 어절(<w>) 단위를 추출하는 기능
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import re


class XMLUnitParser:
    """XML 파일에서 문장과 어절 단위를 추출하는 파서"""
    
    def __init__(self):
        pass
    
    def extract_sentence_units(self, xml_file: str) -> List[Dict[str, Any]]:
        """
        XML 파일에서 문장(<s>) 단위를 추출
        
        Args:
            xml_file: XML 파일 경로
            
        Returns:
            문장 단위 리스트 [{'id': str, 'text': str, 'type': str}, ...]
        """
        try:
            print(f"🔍 XML 파일 경로 확인: {xml_file}")
            print(f"🔍 XML 파일 타입: {type(xml_file)}")
            
            if not Path(xml_file).exists():
                print(f"❌ XML 파일이 존재하지 않습니다: {xml_file}")
                return []
            
            # 다양한 인코딩으로 시도해서 파일 읽기
            xml_content = None
            encodings = ['utf-8-sig', 'utf-8', 'euc-kr', 'cp949', 'latin1']
            
            for encoding in encodings:
                try:
                    with open(xml_file, 'r', encoding=encoding) as f:
                        xml_content = f.read()
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if xml_content is None:
                print(f"❌ 파일 인코딩을 읽을 수 없습니다: {xml_file}")
                return []
            
            # 문자열에서 직접 파싱
            root = ET.fromstring(xml_content)
            
            sentences = []
            seq_sentence_id = 1
            
            # 우선 단락(<단락>) 기준으로 순회하여 문장 단위 추출 (단락/문장 ID 보강)
            found_any = False
            for para_idx, para_elem in enumerate(root.iter('단락'), start=1):
                para_id_attr = para_elem.get('id') or para_elem.get('ID') or para_elem.get('Id')
                para_seq_id = f'P{para_idx:04d}'
                para_id = str(para_id_attr) if para_id_attr else para_seq_id
                
                for s_elem in para_elem.iter('s'):
                    # <s> 내 모든 <w> 텍스트 병합
                    word_texts = []
                    for w_elem in s_elem.iter('w'):
                        word_text = self._extract_text_content(w_elem)
                        if word_text.strip():
                            word_texts.append(word_text.strip())
                    if not word_texts:
                        continue
                    sentence_text = " ".join(word_texts)
                    s_id_attr = s_elem.get('id') or s_elem.get('ID') or s_elem.get('Id')
                    s_id = str(s_id_attr) if s_id_attr else f'S{seq_sentence_id:04d}'
                    
                    sentences.append({
                        'id': s_id,  # 가급적 XML의 s id 사용
                        'sentence_id': s_id,
                        'paragraph_id': para_id,
                        'text': sentence_text,
                        'type': 'sentence',
                        'xml_tag': 's',
                        'word_count': len(word_texts)
                    })
                    seq_sentence_id += 1
                    found_any = True
            
            # 상위 <단락>이 없이 <s> 만 있는 XML도 처리
            if not found_any:
                for s_elem in root.iter('s'):
                    word_texts = []
                    for w_elem in s_elem.iter('w'):
                        word_text = self._extract_text_content(w_elem)
                        if word_text.strip():
                            word_texts.append(word_text.strip())
                    if not word_texts:
                        continue
                    sentence_text = " ".join(word_texts)
                    s_id_attr = s_elem.get('id') or s_elem.get('ID') or s_elem.get('Id')
                    s_id = str(s_id_attr) if s_id_attr else f'S{seq_sentence_id:04d}'
                    sentences.append({
                        'id': s_id,
                        'sentence_id': s_id,
                        'paragraph_id': None,
                        'text': sentence_text,
                        'type': 'sentence',
                        'xml_tag': 's',
                        'word_count': len(word_texts)
                    })
                    seq_sentence_id += 1
            
            # <s> 태그가 없는 경우 단락(<단락>) 내용을 문장으로 처리
            if not sentences:
                for para_elem in root.iter('단락'):
                    text = self._extract_text_content(para_elem)
                    if text.strip():
                        # 문장 부호로 분할
                        split_sentences = self._split_into_sentences(text)
                        for sent_text in split_sentences:
                            if sent_text.strip():
                                sentences.append({
                                    'id': f'S{seq_sentence_id:04d}',
                                    'sentence_id': f'S{seq_sentence_id:04d}',
                                    'paragraph_id': para_elem.get('id') or None,
                                    'text': sent_text.strip(),
                                    'type': 'sentence_from_paragraph',
                                    'xml_tag': '단락'
                                })
                                seq_sentence_id += 1
            
            print(f"📝 문장 단위 추출 완료: {len(sentences)}개 문장")
            return sentences
            
        except Exception as e:
            print(f"❌ 문장 단위 추출 오류: {e}")
            return []
    
    def extract_word_units(self, xml_file: str) -> List[Dict[str, Any]]:
        """
        XML 파일에서 어절(<w>) 단위를 추출
        
        Args:
            xml_file: XML 파일 경로
            
        Returns:
            어절 단위 리스트 [{'id': str, 'text': str, 'type': str}, ...]
        """
        try:
            print(f"🔍 XML 파일 경로 확인: {xml_file}")
            print(f"🔍 XML 파일 타입: {type(xml_file)}")
            
            if not Path(xml_file).exists():
                print(f"❌ XML 파일이 존재하지 않습니다: {xml_file}")
                return []
            
            # 다양한 인코딩으로 시도해서 파일 읽기
            xml_content = None
            encodings = ['utf-8-sig', 'utf-8', 'euc-kr', 'cp949', 'latin1']
            
            for encoding in encodings:
                try:
                    with open(xml_file, 'r', encoding=encoding) as f:
                        xml_content = f.read()
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if xml_content is None:
                print(f"❌ 파일 인코딩을 읽을 수 없습니다: {xml_file}")
                return []
            
            # 문자열에서 직접 파싱
            root = ET.fromstring(xml_content)
            
            words = []
            word_seq = 1
            
            # 단락 → 문장 → 어절 순회하며 문맥 ID 보강
            found_any = False
            for para_idx, para_elem in enumerate(root.iter('단락'), start=1):
                para_id_attr = para_elem.get('id') or para_elem.get('ID') or para_elem.get('Id')
                para_id = str(para_id_attr) if para_id_attr else f'P{para_idx:04d}'
                for s_elem in para_elem.iter('s'):
                    s_id_attr = s_elem.get('id') or s_elem.get('ID') or s_elem.get('Id')
                    s_id = str(s_id_attr) if s_id_attr else None
                    for w_elem in s_elem.iter('w'):
                        text = self._extract_text_content(w_elem)
                        if not text.strip():
                            continue
                        w_id_attr = w_elem.get('id') or w_elem.get('ID') or w_elem.get('Id') or f'w{word_seq}'
                        words.append({
                            'id': f'W{word_seq:04d}',
                            'text': text.strip(),
                            'type': 'word',
                            'xml_tag': 'w',
                            'xml_id': str(w_id_attr),
                            'sentence_id': s_id,  # 상위 문장 ID
                            'paragraph_id': para_id  # 상위 단락 ID
                        })
                        word_seq += 1
                        found_any = True
            
            # 상위 단락/문장 구조없이 <w>만 있는 경우
            if not found_any:
                for w_elem in root.iter('w'):
                    text = self._extract_text_content(w_elem)
                    if not text.strip():
                        continue
                    w_id_attr = w_elem.get('id') or w_elem.get('ID') or w_elem.get('Id') or f'w{word_seq}'
                    words.append({
                        'id': f'W{word_seq:04d}',
                        'text': text.strip(),
                        'type': 'word',
                        'xml_tag': 'w',
                        'xml_id': str(w_id_attr),
                        'sentence_id': None,
                        'paragraph_id': None
                    })
                    word_seq += 1
            
            # <w> 태그가 없는 경우 텍스트를 어절로 분할
            if not words:
                all_text = self._extract_all_text_content(root)
                if all_text:
                    word_tokens = self._split_into_words(all_text)
                    for word_text in word_tokens:
                        if word_text.strip():
                            words.append({
                                'id': f'W{word_id:04d}',
                                'text': word_text.strip(),
                                'type': 'word_from_text',
                                'xml_tag': 'text_split'
                            })
                            word_id += 1
            
            print(f"📝 어절 단위 추출 완료: {len(words)}개 어절")
            return words
            
        except Exception as e:
            print(f"❌ 어절 단위 추출 오류: {e}")
            return []
    
    def _extract_text_content(self, element) -> str:
        """XML 요소에서 텍스트 내용만 추출 (태그 제거)"""
        text_parts = []
        
        # 현재 요소의 텍스트
        if element.text and element.text.strip():
            text_parts.append(element.text.strip())
        
        # 자식 요소들의 텍스트 재귀적으로 추출
        for child in element:
            child_text = self._extract_text_content(child)
            if child_text.strip():
                text_parts.append(child_text.strip())
            
            # tail 텍스트도 포함
            if child.tail and child.tail.strip():
                text_parts.append(child.tail.strip())
        
        # 모든 텍스트 조각을 공백으로 연결
        combined_text = " ".join(text_parts)
        
        # 각주, 원주 등의 참조 번호 제거
        combined_text = re.sub(r'[①-⑳]', '', combined_text)
        combined_text = re.sub(r'\<[^>]*\>', '', combined_text)
        
        # 편집 부호 제거: "[", "]", "-" 문자만 제거
        combined_text = re.sub(r'[\[\-\]]', '', combined_text)  # [, ], - 문자 제거
        
        # 연속된 공백을 하나로 정리
        combined_text = re.sub(r'\s+', ' ', combined_text)
        
        return combined_text
    
    def _join_w_texts(self, word_element) -> str:
        """<w> 태그 내의 모든 텍스트를 결합하여 반환"""
        return self._extract_text_content(word_element)
    
    def _extract_all_text_content(self, root) -> str:
        """전체 XML에서 모든 텍스트 내용 추출"""
        all_text = ""
        
        # 원문과 번역문 영역에서 텍스트 추출
        for elem in root.iter():
            if elem.tag in ['원문', '번역문', '단락']:
                text = self._extract_text_content(elem)
                if text.strip():
                    all_text += " " + text
        
        return all_text
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """텍스트를 문장 단위로 분할"""
        # 한국어와 한문 문장 부호로 분할
        sentence_endings = r'[.!?。？！]\s*'
        sentences = re.split(sentence_endings, text)
        
        # 빈 문자열 제거
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _split_into_words(self, text: str) -> List[str]:
        """텍스트를 어절 단위로 분할"""
        # 공백으로 분할하여 어절 추출
        words = text.split()
        
        # 특수 문자 처리
        processed_words = []
        for word in words:
            # 문장 부호가 붙은 경우 분리
            word_parts = re.findall(r'[\w]+|[^\w\s]', word)
            processed_words.extend(word_parts)
        
        return processed_words
    
    def save_units_to_excel(self, units: List[Dict[str, Any]], output_file: str):
        """단위 데이터를 Excel 파일로 저장"""
        try:
            df = pd.DataFrame(units)
            df.to_excel(output_file, index=False, encoding='utf-8')
            print(f"✅ 단위 데이터 저장 완료: {output_file}")
        except Exception as e:
            print(f"❌ 단위 데이터 저장 오류: {e}")
    
    def extract_sentence_grouped_words(self, xml_file: str) -> List[Dict[str, Any]]:
        """
        XML 파일에서 문장별로 그룹화된 어절 단위를 추출 (SA 분석용)
        
        Returns:
            문장별 그룹화된 구 리스트 [{'sentence_id': str, 'words': [word_dict, ...]}, ...]
        """
        try:
            if not Path(xml_file).exists():
                print(f"❌ XML 파일이 존재하지 않습니다: {xml_file}")
                return []
            
            # XML 파일 읽기
            xml_content = None
            encodings = ['utf-8-sig', 'utf-8', 'euc-kr', 'cp949', 'latin1']
            
            for encoding in encodings:
                try:
                    with open(xml_file, 'r', encoding=encoding) as f:
                        xml_content = f.read()
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if xml_content is None:
                print(f"❌ XML 파일 인코딩 오류: {xml_file}")
                return []
            
            root = ET.fromstring(xml_content)
            sentence_groups = []
            
            # 모든 <s> 태그 찾기
            for sentence_elem in root.findall('.//s'):
                sentence_id = sentence_elem.get('id', f's_{len(sentence_groups)+1}')
                
                # 문장 내 모든 <w> 태그 추출
                words = []
                for word_elem in sentence_elem.findall('.//w'):
                    word_id = word_elem.get('id', f'w_{len(words)+1}')
                    
                    # 어절 텍스트 결합
                    word_text = self._join_w_texts(word_elem)
                    
                    if word_text and word_text.strip():
                        words.append({
                            'id': word_id,
                            'text': word_text,
                            'sentence_id': sentence_id
                        })
                
                if words:  # 어절이 있는 문장만 추가
                    sentence_groups.append({
                        'sentence_id': sentence_id,
                        'words': words,
                        'word_count': len(words)
                    })
            
            print(f"✅ 문장별 어절 그룹 추출 완료: {len(sentence_groups)}개 문장")
            return sentence_groups
            
        except ET.ParseError as e:
            print(f"❌ XML 파싱 오류: {e}")
            return []
        except Exception as e:
            print(f"❌ 문장별 어절 추출 오류: {e}")
            return []
    
    def extract_both_units(self, xml_file: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """문장과 어절 단위를 모두 추출"""
        sentences = self.extract_sentence_units(xml_file)
        words = self.extract_word_units(xml_file)
        
        return sentences, words


def main():
    """테스트 실행"""
    parser = XMLUnitParser()
    
    # 테스트 파일
    test_xml = "c:/Users/junto/Downloads/head-repo/private725/2025/중간버전/관자3_전처리.xml"
    
    if Path(test_xml).exists():
        print(f"🔍 XML 단위 추출 테스트: {test_xml}")
        
        sentences, words = parser.extract_both_units(test_xml)
        
        print(f"\n📊 추출 결과:")
        print(f"  - 문장 단위: {len(sentences)}개")
        print(f"  - 어절 단위: {len(words)}개")
        
        if sentences:
            print(f"\n📝 문장 샘플 (상위 3개):")
            for i, sent in enumerate(sentences[:3]):
                print(f"  {sent['id']}: {sent['text'][:50]}...")
        
        if words:
            print(f"\n📝 어절 샘플 (상위 10개):")
            for i, word in enumerate(words[:10]):
                print(f"  {word['id']}: {word['text']}")
    
    else:
        print(f"❌ 테스트 파일이 존재하지 않습니다: {test_xml}")


if __name__ == "__main__":
    main()