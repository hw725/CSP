"""
XML 프로세서 — XML 원문/번역문에서 문단·문장·구 단위 병렬 데이터 추출

XML 구조:
- 원문 XML: <원문> → <단락> → <s> → <c> → <w>
- 번역문 XML: <번역문> → <단락> → <s> → <c> → <w>

출력 컬럼:
- 문단병렬: 문단식별자, 원문, 번역문
- 문장병렬: 문장식별자, 원문, 번역문
- 구병렬: 구식별자, 원문, 번역문
"""

import re
import pandas as pd
import xml.etree.ElementTree as ET
from typing import List
from pathlib import Path


class XMLPair:
    """XML 쌍을 나타내는 클래스"""
    def __init__(self, pair_id: str = None, name: str = None, original_xml: str = None, translation_xml: str = None, description: str = None, original_path: str = None, translation_path: str = None):
        # 호환성을 위한 매개변수 처리 - original_path/translation_path 우선
        self.original_path = original_path or original_xml
        self.translation_path = translation_path or translation_xml

        # None 체크 추가
        if not self.original_path or not self.translation_path:
            raise ValueError(f"original_path와 translation_path가 모두 필요합니다. original: {self.original_path}, translation: {self.translation_path}")

        # pair_id 생성 - 파일명만 사용, 디렉토리 경로 제거
        if pair_id:
            # pair_id에 경로가 포함되어 있다면 파일명만 추출
            self.id = Path(pair_id).stem if '/' in pair_id or '\\' in pair_id else pair_id
        else:
            orig_name = Path(self.original_path).stem
            # 원문/번역문 공통 부분 추출
            if '원문' in orig_name:
                self.id = orig_name.replace('_원문', '').replace('-원문', '')
            else:
                self.id = f"{orig_name}_{Path(self.translation_path).stem}"

        self.pair_id = self.id  # CLI 호환성을 위한 별칭
        self.name = name or self.id
        self.description = description or ""


def create_xml_pair_from_directory(directory: str, pattern: str = "*원문*.xml") -> List[XMLPair]:
    """디렉토리에서 XML 쌍을 자동으로 찾아 생성"""
    xml_pairs = []
    dir_path = Path(directory)

    # 원문 파일들 찾기
    original_files = list(dir_path.glob("*원문*.xml"))

    for orig_file in original_files:
        # 대응하는 번역문 파일 찾기
        orig_name = orig_file.stem
        # "원문"을 "번역문"으로 바꿔서 매칭 시도
        trans_name = orig_name.replace("원문", "번역문")
        trans_file = dir_path / f"{trans_name}.xml"

        if trans_file.exists():
            xml_pairs.append(XMLPair(original_path=str(orig_file), translation_path=str(trans_file)))

    return xml_pairs


class XMLProcessor:
    """올바른 XML 구조를 처리하는 XML 프로세서"""

    @staticmethod
    def _extract_w_texts(element) -> List[str]:
        """요소에서 모든 w 태그의 텍스트를 순서대로 추출 (전처리 포함)"""
        if element is None:
            return []

        w_elements = element.findall('.//w')
        texts = []

        for w in w_elements:
            w_text = (w.text or "").strip()
            if w_text:
                # [ ] - 문자 제거
                w_text = re.sub(r'[\[\-\]]', '', w_text)
                # 연속된 공백 정리
                w_text = re.sub(r'\s+', ' ', w_text).strip()

                if w_text:  # 전처리 후에도 텍스트가 남아있으면 추가
                    texts.append(w_text)

        return texts

    @staticmethod
    def _join_w_texts(element) -> str:
        """요소에서 모든 w 태그의 텍스트를 공백으로 연결 (전처리 포함)"""
        texts = XMLProcessor._extract_w_texts(element)
        combined = " ".join(texts)
        # 추가 전처리: [, ], - 문자 제거 및 연속된 공백 정리
        combined = re.sub(r'[\[\-\]]', '', combined)  # [, ], - 문자 제거
        combined = re.sub(r'\s+', ' ', combined).strip()
        return combined

    @staticmethod
    def extract_paragraph_data(orig_xml: str, trans_xml: str) -> pd.DataFrame:
        """XML 쌍에서 문단병렬 데이터 추출"""
        try:
            print(f"[extract] 문단병렬 추출:")
            print(f"   원문: {Path(orig_xml).name}")
            print(f"   번역: {Path(trans_xml).name}")

            # 원문 XML 파싱
            orig_tree = ET.parse(orig_xml)
            orig_root = orig_tree.getroot()

            # 번역문 XML 파싱
            trans_tree = ET.parse(trans_xml)
            trans_root = trans_tree.getroot()

            data = []

            # 원문에서 원문 요소들 추출 (네 가지 구조 지원)
            orig_elements = orig_root.findall('.//원문')
            if not orig_elements:
                orig_elements = orig_root.findall('.//단락[@type="P"]')
            if not orig_elements:
                orig_elements = orig_root.findall('.//경문원문')
            if not orig_elements:
                orig_elements = orig_root.findall('.//단락[@id]')
            print(f"   원문 요소: {len(orig_elements)}개")

            # 번역문에서 번역문 요소들 추출 (네 가지 구조 지원)
            trans_elements = trans_root.findall('.//번역문')
            if not trans_elements:
                trans_elements = trans_root.findall('.//단락[@type="T"]')
            if not trans_elements:
                trans_elements = trans_root.findall('.//경문번역')
            if not trans_elements:
                trans_elements = trans_root.findall('.//단락[@id]')
            print(f"   번역문 요소: {len(trans_elements)}개")

            # ID 기반 매칭 (식별자 속성 사용)
            pair_id = 1

            for orig_elem in orig_elements:
                # 여러 가능한 ID 속성 확인
                orig_id = orig_elem.get('식별자') or orig_elem.get('id')

                if orig_id:  # ID가 있는 경우만 처리
                    # 같은 ID를 가진 번역문 찾기 (ID:W1 -> ID:W1_T 매칭)
                    matching_trans = None
                    for trans_elem in trans_elements:
                        trans_id = trans_elem.get('식별자') or trans_elem.get('id')
                        if trans_id == f"{orig_id}_T" or trans_id == orig_id:
                            matching_trans = trans_elem
                            break
                else:
                    # ID가 없으면 순서 기반 매칭 (위치 기반)
                    if pair_id <= len(trans_elements):
                        matching_trans = trans_elements[pair_id - 1]
                    else:
                        matching_trans = None

                if matching_trans is not None:
                    orig_text = XMLProcessor._join_w_texts(orig_elem)
                    trans_text = XMLProcessor._join_w_texts(matching_trans)

                    if orig_text.strip() and trans_text.strip():
                        data.append({
                            '문단식별자': orig_id,
                            '원문': orig_text.strip(),
                            '번역문': trans_text.strip()
                        })

                pair_id += 1

            print(f"   -> {len(data)}개 문단 쌍 추출")
            return pd.DataFrame(data)

        except Exception as e:
            print(f"[ERROR] XML 문단 추출 오류: {e}")
            raise Exception(f"XML 문단 추출 실패: {e}")

    @staticmethod
    def extract_sentence_data(orig_xml: str, trans_xml: str) -> pd.DataFrame:
        """XML 쌍에서 문장병렬 데이터 추출"""
        try:
            print(f"[extract] 문장병렬 추출:")
            print(f"   원문: {Path(orig_xml).name}")
            print(f"   번역: {Path(trans_xml).name}")

            # XML 파싱
            orig_tree = ET.parse(orig_xml)
            orig_root = orig_tree.getroot()

            trans_tree = ET.parse(trans_xml)
            trans_root = trans_tree.getroot()

            data = []

            # 원문과 번역문 요소들 (s 요소 = 문장 단위)
            orig_elements = orig_root.findall('.//s')
            trans_elements = trans_root.findall('.//s')

            sentence_id = 1

            # s 요소별 매칭 (문장 단위)
            for orig_s, trans_s in zip(orig_elements, trans_elements):
                orig_text = (orig_s.text or "").strip()
                trans_text = (trans_s.text or "").strip()

                # s 요소 내부의 모든 텍스트 수집
                if not orig_text:
                    orig_text = "".join(orig_s.itertext()).strip()
                if not trans_text:
                    trans_text = "".join(trans_s.itertext()).strip()

                # -, [, ] 문자 제거
                if orig_text:
                    orig_text = orig_text.replace('-', '').replace('[', '').replace(']', '')
                if trans_text:
                    trans_text = trans_text.replace('-', '').replace('[', '').replace(']', '')

                if orig_text and trans_text:
                    data.append({
                        '문장식별자': sentence_id,
                        '원문': orig_text,
                        '번역문': trans_text
                    })
                    sentence_id += 1

            print(f"   -> {len(data)}개 문장 쌍 추출")
            return pd.DataFrame(data)

        except Exception as e:
            print(f"[ERROR] XML 문장 추출 오류: {e}")
            raise Exception(f"XML 문장 추출 실패: {e}")

    @staticmethod
    def extract_phrase_data(orig_xml: str, trans_xml: str) -> pd.DataFrame:
        """XML 쌍에서 구병렬 데이터 추출"""
        try:
            print(f"[extract] 구병렬 추출:")
            print(f"   원문: {Path(orig_xml).name}")
            print(f"   번역: {Path(trans_xml).name}")

            # XML 파싱
            orig_tree = ET.parse(orig_xml)
            orig_root = orig_tree.getroot()

            trans_tree = ET.parse(trans_xml)
            trans_root = trans_tree.getroot()

            data = []

            # 먼저 원문/번역문 요소가 있는지 확인 (한유2, 한유3, 구양수6 방식)
            orig_elements = orig_root.findall('.//원문')
            trans_elements = trans_root.findall('.//번역문')

            phrase_id = 1

            if orig_elements and trans_elements:
                # 한유2, 한유3, 구양수6 방식 (원문/번역문 요소 사용)
                for orig_elem in orig_elements:
                    orig_id = orig_elem.get('식별자')

                    # 매칭되는 번역문 찾기 (ID:W1 -> ID:W1_T)
                    matching_trans = None
                    if orig_id:
                        target_trans_id = orig_id + "_T"
                        for trans_elem in trans_elements:
                            if trans_elem.get('식별자') == target_trans_id:
                                matching_trans = trans_elem
                                break

                    if matching_trans is not None:
                        # w 요소들 추출 (구 역할)
                        orig_w_elements = orig_elem.findall('.//w')
                        trans_w_elements = matching_trans.findall('.//w')

                        # w 요소별 매칭 (구 단위)
                        for orig_w, trans_w in zip(orig_w_elements, trans_w_elements):
                            orig_text = (orig_w.text or "").strip()
                            trans_text = (trans_w.text or "").strip()

                            if orig_text and trans_text:
                                data.append({
                                    '구식별자': phrase_id,
                                    '원문': orig_text,
                                    '번역문': trans_text
                                })
                                phrase_id += 1
            else:
                # 대학장구 방식 (w 요소 직접 사용)
                orig_w_elements = orig_root.findall('.//w')
                trans_w_elements = trans_root.findall('.//w')

                # ID 기반 매칭
                for orig_w in orig_w_elements:
                    orig_id = orig_w.get('id')
                    orig_text = (orig_w.text or "").strip()

                    if orig_id and orig_text:
                        # 같은 ID를 가진 번역문 w 요소 찾기
                        matching_trans_w = None
                        for trans_w in trans_w_elements:
                            if trans_w.get('id') == orig_id:
                                matching_trans_w = trans_w
                                break

                        if matching_trans_w is not None:
                            trans_text = (matching_trans_w.text or "").strip()
                            if trans_text:
                                # -, [, ] 문자 제거
                                orig_text_clean = orig_text.replace('-', '').replace('[', '').replace(']', '')
                                trans_text_clean = trans_text.replace('-', '').replace('[', '').replace(']', '')

                                data.append({
                                    '구식별자': phrase_id,
                                    '원문': orig_text_clean,
                                    '번역문': trans_text_clean
                                })
                                phrase_id += 1

            print(f"   -> {len(data)}개 구 쌍 추출")
            return pd.DataFrame(data)

        except Exception as e:
            print(f"[ERROR] XML 구 추출 오류: {e}")
            raise Exception(f"XML 구 추출 실패: {e}")
