#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PowerShell 인코딩 손상 역추적 및 복구
"""
import sys

# 원래: UTF-8 한글 파일
# PowerShell: Get-Content -> 문자열로 읽음 -> Set-Content (기본 인코딩)
# 문제: PowerShell이 UTF-8 파일을 잘못된 인코딩으로 읽거나 저장

# 테스트
original = '전근대'
orig_bytes = original.encode('utf-8')
print(f'Original: {original}')
print(f'Original UTF-8 bytes: {orig_bytes.hex()}')

# UTF-8 바이트를 CP949로 해석하면?
try:
    cp949_decoded = orig_bytes.decode('cp949', errors='replace')
    print(f'UTF-8 bytes decoded as CP949: {cp949_decoded}')
    # 다시 UTF-8로 저장
    back = cp949_decoded.encode('utf-8')
    print(f'Back to UTF-8: {back.hex()}')
except Exception as e:
    print(f'Error: {e}')

print('\n--- Analyzing corrupted file ---')

with open('CLASSIFIED_MARKERS.md', 'rb') as f:
    raw = f.read()

# 제어문자 제거
if raw[0] == 0x12:
    raw = raw[1:]
    print('Removed leading control char 0x12')

print(f'File size: {len(raw)} bytes')
print(f'First 100 bytes hex: {raw[:100].hex()}')

# 현재 파일은 UTF-8로 저장되어 있음
# 손상 과정 추정:
# 1. 원본 UTF-8 파일
# 2. PowerShell Get-Content -Encoding UTF8 으로 읽음 (정상)
# 3. Set-Content 기본 인코딩으로 저장 (문제!)
# 4. 시스템 기본 인코딩(CP949)이 적용되어 바이트 시퀀스 변경

# 역변환 시도:
# 현재 파일(UTF-8) -> UTF-8 디코딩 -> CP949 인코딩 -> UTF-8 디코딩

text = raw.decode('utf-8', errors='surrogateescape')
print(f'\nDecoded as UTF-8 (first 200 chars):')
print(text[:200])

# 이 문자열을 CP949 바이트로 인코딩 -> 원래 UTF-8 바이트로 해석
try:
    cp949_bytes = text.encode('cp949', errors='surrogateescape')
    recovered = cp949_bytes.decode('utf-8', errors='replace')
    print(f'\nRecovered (CP949 encode -> UTF-8 decode):')
    print(recovered[:500])
except Exception as e:
    print(f'Recovery failed: {e}')

# 다른 시도: latin-1 경유
try:
    latin_bytes = text.encode('latin-1', errors='surrogateescape')
    recovered2 = latin_bytes.decode('utf-8', errors='replace')
    print(f'\nRecovered via latin-1:')
    print(recovered2[:500])
except Exception as e:
    print(f'Latin-1 recovery failed: {e}')
