# -*- coding: utf-8 -*-
"""
인천-서울 광역통근 최적 출퇴근 골든타임 예측 및 추천 전체 실행 파이프라인
실행 방법: python .\incheon-seoul-commute-golden-time_main.py
"""

import os
import sys
import importlib.util


def _register_prefixed_modules():
    """src 폴더 내 접두사(incheon-seoul-commute-golden-time_src_)가 붙은 모듈을 표준 경로로 자동 매핑"""
    prefix = "incheon-seoul-commute-golden-time_src_"
    modules = ["data_loader", "preprocess", "train_evaluate", "eda_visualization", "recommend"]
    for mod in modules:
        prefixed_path = os.path.join("src", f"{prefix}{mod}.py")
        normal_path = os.path.join("src", f"{mod}.py")
        target_path = prefixed_path if os.path.exists(prefixed_path) else normal_path
        if os.path.exists(target_path):
            spec = importlib.util.spec_from_file_location(f"src.{mod}", target_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"src.{mod}"] = module
            spec.loader.exec_module(module)


_register_prefixed_modules()

from src.data_loader import load_or_create_raw_data
from src.preprocess import run_preprocessing
from src.train_evaluate import train_and_evaluate
from src.eda_visualization import generate_eda_and_charts
from src.recommend import build_golden_time_recommendations


def main():
    print("=" * 72)
    print(" [STEP 1] 공공데이터 원천 구조(Wide-Format) 로딩")
    print("=" * 72)
    prefixed_raw = "data/incheon-seoul-commute-golden-time_data_raw_public_transport_wide.csv"
    raw_path = prefixed_raw if os.path.exists(prefixed_raw) else "data/raw_public_transport_wide.csv"
    df_raw = load_or_create_raw_data(raw_path)
    print(f" - 원천 데이터 크기: {df_raw.shape[0]}행 x {df_raw.shape[1]}열")

    print("\n" + "=" * 72)
    print(" [STEP 2] 데이터 전처리 (Melt, 문자열 정제, 결측치 중앙값 보간, 파생변수)")
    print("=" * 72)
    df_clean, prep_summary = run_preprocessing(df_raw, "data/preprocessed_incheon_seoul_commute.csv")
    print(f" - 전처리 완료 크기: {prep_summary['clean_shape'][0]:,}행 x {prep_summary['clean_shape'][1]}열")
    print(f" - '%' 기호 정제: {prep_summary['str_pct_cleaned']}건 | 결측 보간: {prep_summary['missing_before']}건 -> {prep_summary['missing_after']}건")
    print(f" - 쾌적등급 분포: {prep_summary['class_distribution']}")

    print("\n" + "=" * 72)
    print(" [STEP 3] 머신러닝 5종 분류 및 회귀 모델 학습·평가")
    print("=" * 72)
    df_results, df_fi, reg_metrics, best_cls, reg_model = train_and_evaluate(df_clean, "outputs")
    print(df_results.to_string(index=False))
    print(f"\n - 회귀 모델(Random Forest Regressor): RMSE={reg_metrics['RMSE']}%p, R2={reg_metrics['R2']}")
    print("\n - 주요 영향 요인 Top 5:")
    print(df_fi.head(5).to_string(index=False))

    print("\n" + "=" * 72)
    print(" [STEP 4] 탐색적 데이터 분석(EDA) 및 시각화 6종 생성")
    print("=" * 72)
    generate_eda_and_charts(df_clean, df_results, df_fi, "outputs/figures")
    print(" - outputs/figures/ 경로에 6종 고해상도 차트 저장 완료")

    print("\n" + "=" * 72)
    print(" [STEP 5] 출근(Part 1) & 퇴근(Part 2) 쾌적 골든타임 추천표 산출")
    print("=" * 72)
    df_rec = build_golden_time_recommendations("outputs")
    print(df_rec.to_string(index=False))
    print("\n전체 파이프라인 실행이 완료되었습니다.")


if __name__ == "__main__":
    main()