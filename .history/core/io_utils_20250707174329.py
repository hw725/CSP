"""
IO 유틸리티 모듈 - 파일 입출력 및 데이터 관리
"""
import pandas as pd
import os
from typing import List, Dict, Any, Optional
import logging

class IOManager:
    """파일 입출력 관리 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def read_excel(self, file_path: str, sheet_name: str = None) -> pd.DataFrame:
        """Excel 파일 읽기"""
        try:
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path)
            self.logger.info(f"Excel 파일 읽기 성공: {file_path}, 행 수: {len(df)}")
            return df
        except Exception as e:
            self.logger.error(f"Excel 파일 읽기 실패: {file_path}, 오류: {e}")
            raise
    
    def write_excel(self, df: pd.DataFrame, file_path: str, sheet_name: str = 'Sheet1') -> None:
        """Excel 파일 쓰기"""
        try:
            df.to_excel(file_path, sheet_name=sheet_name, index=False)
            self.logger.info(f"Excel 파일 쓰기 성공: {file_path}, 행 수: {len(df)}")
        except Exception as e:
            self.logger.error(f"Excel 파일 쓰기 실패: {file_path}, 오류: {e}")
            raise
    
    def validate_file_exists(self, file_path: str) -> bool:
        """파일 존재 여부 확인"""
        exists = os.path.exists(file_path)
        if not exists:
            self.logger.warning(f"파일이 존재하지 않음: {file_path}")
        return exists
    
    def create_directory(self, dir_path: str) -> None:
        """디렉토리 생성"""
        os.makedirs(dir_path, exist_ok=True)
        self.logger.info(f"디렉토리 생성/확인: {dir_path}")
    
    def backup_file(self, file_path: str) -> str:
        """파일 백업"""
        if not self.validate_file_exists(file_path):
            return ""
        
        backup_path = f"{file_path}.backup"
        try:
            import shutil
            shutil.copy2(file_path, backup_path)
            self.logger.info(f"파일 백업 완료: {file_path} -> {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.error(f"파일 백업 실패: {file_path}, 오류: {e}")
            return ""
