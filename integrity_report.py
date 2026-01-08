#!/usr/bin/env python3
"""전체 43권 무결성(텍스트 길이) 리포트 생성"""

import pandas as pd
from pathlib import Path
import json

books = [
    "예기집설대전1", "예기집설대전2",
    "춘추좌씨전1", "춘추좌씨전2", "춘추좌씨전3", "춘추좌씨전4", 
    "춘추좌씨전5", "춘추좌씨전6", "춘추좌씨전7", "춘추좌씨전8",
    "자치통감강목1", "자치통감강목2", "자치통감강목3", "자치통감강목4",
    "자치통감강목5", "자치통감강목6", "자치통감강목7",
    "당시삼백수1", "당시삼백수2", "당시삼백수3",
    "당송팔대가문초한유1", "당송팔대가문초한유2", "당송팔대가문초한유3",
    "당송팔대가문초유종원1", "당송팔대가문초유종원2",
    "당송팔대가문초구양수1", "당송팔대가문초구양수2", "당송팔대가문초구양수3",
    "당송팔대가문초구양수4", "당송팔대가문초구양수5", "당송팔대가문초구양수6",
    "당송팔대가문초소순1",
    "당송팔대가문초소식1", "당송팔대가문초소식2", "당송팔대가문초소식3",
    "당송팔대가문초소식4", "당송팔대가문초소식5",
    "당송팔대가문초소철1", "당송팔대가문초소철2", "당송팔대가문초소철3",
    "당송팔대가문초왕안석1", "당송팔대가문초왕안석2",
    "당송팔대가문초증공1",
]

results = []

for book in books:
    # 입력: 문단병렬
    para_file = Path(f"xlsx/{book}/{book}_문단병렬.xlsx")
    # PA 출력
    pa_file = Path(f"xlsx_pipeline_results/{book}/{book}_PA_문장병렬.xlsx")
    # GT: 문장병렬
    gt_file = Path(f"xlsx/{book}/{book}_문장병렬.xlsx")
    # SA 출력
    sa_file = Path(f"xlsx_pipeline_results/{book}/{book}_SA.xlsx")
    
    record = {"책": book}
    
    # 입력 (문단병렬)
    if para_file.exists():
        para = pd.read_excel(para_file)
        record["입력_행"] = len(para)
        record["입력_원문_길이"] = para["원문"].astype(str).map(len).sum()
        record["입력_번역_길이"] = para["번역문"].astype(str).map(len).sum()
    else:
        record["입력_행"] = None
        record["입력_원문_길이"] = None
        record["입력_번역_길이"] = None
    
    # PA 출력
    if pa_file.exists():
        pa = pd.read_excel(pa_file)
        record["PA_행"] = len(pa)
        record["PA_원문_길이"] = pa["원문"].astype(str).map(len).sum()
        record["PA_번역_길이"] = pa["번역문"].astype(str).map(len).sum()
    else:
        record["PA_행"] = None
        record["PA_원문_길이"] = None
        record["PA_번역_길이"] = None
    
    # GT (문장병렬)
    if gt_file.exists():
        gt = pd.read_excel(gt_file)
        record["GT_행"] = len(gt)
        record["GT_원문_길이"] = gt["원문"].astype(str).map(len).sum()
        record["GT_번역_길이"] = gt["번역문"].astype(str).map(len).sum()
    else:
        record["GT_행"] = None
        record["GT_원문_길이"] = None
        record["GT_번역_길이"] = None
    
    # SA 출력
    if sa_file.exists():
        try:
            sa = pd.read_excel(sa_file)
            record["SA_행"] = len(sa)
            record["SA_원문_길이"] = sa["원문"].astype(str).map(len).sum()
            record["SA_번역_길이"] = sa["번역문"].astype(str).map(len).sum()
        except Exception as e:
            record["SA_행"] = None
            record["SA_원문_길이"] = None
            record["SA_번역_길이"] = None
    else:
        record["SA_행"] = None
        record["SA_원문_길이"] = None
        record["SA_번역_길이"] = None
    
    # 무결성 체크: PA vs 입력
    if record["입력_원문_길이"] and record["PA_원문_길이"]:
        record["PA_원문_Δ"] = record["PA_원문_길이"] - record["입력_원문_길이"]
        record["PA_번역_Δ"] = record["PA_번역_길이"] - record["입력_번역_길이"]
    else:
        record["PA_원문_Δ"] = None
        record["PA_번역_Δ"] = None
    
    # 무결성 체크: SA vs 입력
    if record["입력_원문_길이"] and record["SA_원문_길이"]:
        record["SA_원문_Δ"] = record["SA_원문_길이"] - record["입력_원문_길이"]
        record["SA_번역_Δ"] = record["SA_번역_길이"] - record["입력_번역_길이"]
    else:
        record["SA_원문_Δ"] = None
        record["SA_번역_Δ"] = None
    
    results.append(record)

# 저장
df = pd.DataFrame(results)
df.to_csv("analytics/무결성_리포트.csv", index=False, encoding="utf-8-sig")

# 콘솔 출력
print("="*120)
print("📊 전체 43권 무결성 리포트")
print("="*120)
print(df.to_string(index=False))

# 요약 통계
print("\n" + "="*120)
print("📈 요약 통계")
print("="*120)

# PA 무결성 이상 (Δ ≠ 0)
pa_issues = df[df["PA_원문_Δ"].notna() & (df["PA_원문_Δ"] != 0)]
print(f"\n🔴 PA 원문 무결성 문제 (길이 변형):")
print(f"   문제 책: {len(pa_issues)}권")
if len(pa_issues) > 0:
    print(pa_issues[["책", "PA_원문_Δ", "PA_번역_Δ"]].to_string(index=False))

# SA 무결성 이상
sa_issues = df[df["SA_원문_Δ"].notna() & (df["SA_원문_Δ"] != 0)]
print(f"\n🔴 SA 원문 무결성 문제 (길이 변형):")
print(f"   문제 책: {len(sa_issues)}권")
if len(sa_issues) > 0:
    print(sa_issues[["책", "SA_원문_Δ", "SA_번역_Δ"]].to_string(index=False))

# 분할 개수 차이 (PA vs GT)
df["분할_차이_행"] = df["PA_행"] - df["GT_행"]
split_issues = df[df["분할_차이_행"].notna() & (df["분할_차이_행"] != 0)]
print(f"\n⚠️  분할 개수 차이 (PA 행 수 vs GT 행 수):")
print(f"   차이 있는 책: {len(split_issues)}권")
if len(split_issues) > 0:
    print(split_issues[["책", "PA_행", "GT_행", "분할_차이_행"]].to_string(index=False))

print("\n✅ 리포트 저장: analytics/무결성_리포트.csv")
