# -*- coding: utf-8 -*-
"""
데이터 전처리 파이프라인 모듈 (pandas)
1) Wide -> Long 구조 변환 (pd.melt)
2) 문자열('%') 정제 및 수치형(float) 변환
3) [노선명 x 방향 x 출발시간대] 그룹별 중앙값 결측치 보간
4) 파생변수 생성 (시간_수치, 평일여부, 강수여부, 통근구간구분)
5) 타겟 변수(Y) 정의: 3단계 쾌적등급 (0: 쾌적[<=80%], 1: 보통[80~115%], 2: 혼잡[>115%])
"""

import os
import numpy as np
import pandas as pd
from src.data_loader import time_to_float


def run_preprocessing(df_raw: pd.DataFrame, output_csv_path: str = "data/preprocessed_incheon_seoul_commute.csv"):
    time_slots = [c for c in df_raw.columns if '시' in c and '분' in c]
    id_vars = [c for c in df_raw.columns if c not in time_slots]

    # 1. Wide -> Long 변환
    df_long = df_raw.melt(
        id_vars=id_vars,
        value_vars=time_slots,
        var_name='출발시간대',
        value_name='혼잡도_원천'
    )

    missing_before = int(df_long['혼잡도_원천'].isna().sum())
    str_pct_count = int(df_long['혼잡도_원천'].astype(str).str.contains('%').sum())

    # 2. '%' 문자열 제거 및 수치 변환
    df_long['혼잡도'] = (
        df_long['혼잡도_원천']
        .astype(str)
        .str.replace('%', '', regex=False)
        .replace('nan', np.nan)
        .astype(float)
    )

    # 3. 그룹별 중앙값 기반 결측치 보간
    df_long['혼잡도'] = df_long['혼잡도'].fillna(
        df_long.groupby(['호선_노선명', '상하_방향구분', '출발시간대'])['혼잡도'].transform('median')
    )
    missing_after = int(df_long['혼잡도'].isna().sum())

    # 4. 파생변수 생성
    df_long['시간_수치'] = df_long['출발시간대'].apply(time_to_float)
    df_long['평일여부'] = df_long['요일구분'].apply(lambda x: 1 if str(x).startswith('평일') else 0)
    df_long['강수여부'] = df_long['기상상태'].apply(lambda x: 1 if x == '비_눈' else 0)

    def _get_commute_window(row):
        t, d = row['시간_수치'], row['상하_방향구분']
        if d == '서울방면' and 6.0 <= t <= 10.0:
            return '오전출근_핵심구간'
        elif d == '인천방면' and 17.0 <= t <= 21.5:
            return '오후퇴근_핵심구간'
        return '비첨두_일반구간'

    df_long['통근구간구분'] = df_long.apply(_get_commute_window, axis=1)

    # 5. 타겟 변수(Y) 정의
    def _classify_comfort(c):
        if c <= 80.0:
            return 0  # 쾌적
        elif c <= 115.0:
            return 1  # 보통
        return 2      # 혼잡

    df_long['쾌적등급'] = df_long['혼잡도'].apply(_classify_comfort)
    comfort_map = {0: '쾌적(≤80%)', 1: '보통(80~115%)', 2: '혼잡(>115%)'}
    df_long['쾌적등급명'] = df_long['쾌적등급'].map(comfort_map)

    drop_cols = [c for c in ['연번', '혼잡도_원천'] if c in df_long.columns]
    df_clean = df_long.drop(columns=drop_cols).copy()

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_clean.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    summary = {
        "raw_shape": df_raw.shape,
        "clean_shape": df_clean.shape,
        "str_pct_cleaned": str_pct_count,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "class_distribution": df_clean['쾌적등급명'].value_counts().to_dict()
    }
    return df_clean, summary
