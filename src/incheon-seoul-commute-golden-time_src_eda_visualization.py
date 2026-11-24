# -*- coding: utf-8 -*-
"""
탐색적 데이터 분석(EDA) 및 고해상도 시각화 모듈 (Matplotlib & Seaborn)
- 노선별·방향별 기술통계량 및 변수 간 상관계수 분석
- Part 1(오전 출근: 인천->서울) & Part 2(오후 퇴근: 서울->인천) 분리 시각화
- 기상 악화(비·눈) 영향 및 머신러닝 모델 성능/중요도 시각화
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from src.data_loader import TIME_SLOTS

ORDERED_LINES = ['1호선', '7호선', '공항철도', 'M6405', 'M6724', 'M6117', '1400번', '9500번']


def setup_korean_font():
    candidates = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        'C:/Windows/Fonts/malgun.ttf',
        '/System/Library/Fonts/AppleSDGothicNeo.ttc'
    ]
    for path in candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            prop = fm.FontProperties(fname=path)
            plt.rcParams['font.family'] = prop.get_name()
            break
    plt.rcParams['axes.unicode_minus'] = False


def generate_eda_and_charts(df_clean: pd.DataFrame, df_results: pd.DataFrame, df_fi: pd.DataFrame, fig_dir: str = "outputs/figures"):
    setup_korean_font()
    os.makedirs(fig_dir, exist_ok=True)

    # 1. 그림 1: 양방향 시간대별 혼잡도 히트맵
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    hm_seoul = df_clean[(df_clean['평일여부'] == 1) & (df_clean['상하_방향구분'] == '서울방면')].pivot_table(
        index='호선_노선명', columns='출발시간대', values='혼잡도', aggfunc='mean'
    ).reindex(ORDERED_LINES)[TIME_SLOTS]

    hm_incheon = df_clean[(df_clean['평일여부'] == 1) & (df_clean['상하_방향구분'] == '인천방면')].pivot_table(
        index='호선_노선명', columns='출발시간대', values='혼잡도', aggfunc='mean'
    ).reindex(ORDERED_LINES)[TIME_SLOTS]

    sns.heatmap(hm_seoul, ax=axes[0], cmap='YlOrRd', vmin=25, vmax=145, cbar_kws={'label': '혼잡도(%)'})
    axes[0].set_title('평일 [서울방면(상행)] 노선별·시간대별 평균 혼잡도 히트맵 (인천 → 서울)', fontsize=13, fontweight='bold', pad=10)
    axes[0].set_ylabel('노선 / 광역버스')

    sns.heatmap(hm_incheon, ax=axes[1], cmap='YlOrRd', vmin=25, vmax=145, cbar_kws={'label': '혼잡도(%)'})
    axes[1].set_title('평일 [인천방면(하행)] 노선별·시간대별 평균 혼잡도 히트맵 (서울 → 인천)', fontsize=13, fontweight='bold', pad=10)
    axes[1].set_ylabel('노선 / 광역버스')
    axes[1].set_xlabel('출발 시간대 (05:30 ~ 23:30)')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/fig1_eda_heatmap_corridors.png", dpi=200)
    plt.close()

    # 2. 그림 2: 주요 수치형·이진 변수 간 상관계수 히트맵
    corr_df = df_clean[['혼잡도', '쾌적등급', '시간_수치', '환승거점여부', '평일여부', '강수여부']].copy()
    corr_df['출근피크_지시'] = ((df_clean['상하_방향구분'] == '서울방면') & (df_clean['시간_수치'].between(7.5, 8.5))).astype(int)
    corr_df['퇴근피크_지시'] = ((df_clean['상하_방향구분'] == '인천방면') & (df_clean['시간_수치'].between(18.0, 19.5))).astype(int)
    corr_df['광역버스여부'] = (df_clean['수단구분'] == '광역버스').astype(int)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_df.corr().round(2), annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('주요 통근 변수 및 혼잡도 간 상관계수 행렬 (Correlation Matrix)', fontsize=12.5, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/fig2_correlation_matrix.png", dpi=200)
    plt.close()

    # 3. 그림 3: [Part 1] 오전 출근 시간대(인천 -> 서울) 추이 및 시프트 효과
    am_df = df_clean[(df_clean['평일여부'] == 1) & (df_clean['강수여부'] == 0) & (df_clean['상하_방향구분'] == '서울방면')]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    highlight_lines = ['1호선', '7호선', '공항철도', 'M6405', 'M6117']
    colors_am = ['#1f77b4', '#2ca02c', '#9467bd', '#d62728', '#ff7f0e']

    am_curve = df_clean[(df_clean['평일여부'] == 1) & (df_clean['상하_방향구분'] == '서울방면') & (df_clean['시간_수치'].between(5.5, 11.0))]
    for line, col in zip(highlight_lines, colors_am):
        s = am_curve[am_curve['호선_노선명'] == line].groupby('시간_수치')['혼잡도'].mean()
        axes[0].plot(s.index, s.values, marker='o', markersize=4, linewidth=2.2, label=line, color=col)

    axes[0].axhspan(0, 80, color='#27ae60', alpha=0.13, label='쾌적 구간 (≤80%: 착석/여유)')
    axes[0].axvspan(7.5, 8.5, color='#e74c3c', alpha=0.10, label='출근 최악 피크 (07:30~08:30)')
    axes[0].axhline(115, color='#c0392b', linestyle='--', linewidth=1.2, label='혼잡 경보선 (115%)')
    axes[0].set_title('[PART 1-A] 오전 출근(인천→서울) 시간대별 혼잡도 추이', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('출발 시각 (시)')
    axes[0].set_ylabel('평균 혼잡도 (%)')
    axes[0].set_xticks([6, 7, 8, 9, 10, 11])
    axes[0].set_xticklabels(['06:00', '07:00', '08:00', '09:00', '10:00', '11:00'])
    axes[0].legend(fontsize=8.5, loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    am_shifts = ['06시30분', '07시00분', '08시00분', '09시00분', '09시30분']
    am_bar = am_df[am_df['출발시간대'].isin(am_shifts)].groupby(['호선_노선명', '출발시간대'])['혼잡도'].mean().unstack()[am_shifts].reindex(ORDERED_LINES)
    am_bar.plot(kind='bar', ax=axes[1], width=0.78, color=['#1e8449', '#2ecc71', '#e74c3c', '#f39c12', '#3498db'])
    axes[1].axhline(80, color='#1e8449', linestyle='--', linewidth=1.3, label='쾌적 기준선 (80%)')
    axes[1].axhline(115, color='#c0392b', linestyle=':', linewidth=1.3, label='혼잡 경보선 (115%)')
    axes[1].set_title('[PART 1-B] 노선별 출근 골든타임 시프트 효과 (08:00 회피)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('노선 / 광역버스')
    axes[1].set_ylabel('평균 혼잡도 (%)')
    axes[1].tick_params(axis='x', rotation=25)
    axes[1].legend(fontsize=8, loc='upper right', ncol=2)
    axes[1].grid(True, axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/fig3_part1_morning_commute.png", dpi=200)
    plt.close()

    # 4. 그림 4: [Part 2] 오후 퇴근 시간대(서울 -> 인천) 추이 및 시프트 효과
    pm_df = df_clean[(df_clean['평일여부'] == 1) & (df_clean['강수여부'] == 0) & (df_clean['상하_방향구분'] == '인천방면')]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))

    pm_curve = df_clean[(df_clean['평일여부'] == 1) & (df_clean['상하_방향구분'] == '인천방면') & (df_clean['시간_수치'].between(16.5, 22.0))]
    for line, col in zip(highlight_lines, colors_am):
        s = pm_curve[pm_curve['호선_노선명'] == line].groupby('시간_수치')['혼잡도'].mean()
        axes[0].plot(s.index, s.values, marker='o', markersize=4, linewidth=2.2, label=line, color=col)

    axes[0].axhspan(0, 80, color='#27ae60', alpha=0.13, label='쾌적 구간 (≤80%: 착석/여유)')
    axes[0].axvspan(18.0, 19.2, color='#e74c3c', alpha=0.10, label='퇴근 최악 피크 (18:00~19:15)')
    axes[0].axhline(115, color='#c0392b', linestyle='--', linewidth=1.2, label='혼잡 경보선 (115%)')
    axes[0].set_title('[PART 2-A] 오후 퇴근(서울→인천) 시간대별 혼잡도 추이', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('출발 시각 (시)')
    axes[0].set_ylabel('평균 혼잡도 (%)')
    axes[0].set_xticks([17, 18, 19, 20, 21, 22])
    axes[0].set_xticklabels(['17:00', '18:00', '19:00', '20:00', '21:00', '22:00'])
    axes[0].legend(fontsize=8.5, loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    pm_shifts = ['17시00분', '17시30분', '18시30분', '19시30분', '20시00분']
    pm_bar = pm_df[pm_df['출발시간대'].isin(pm_shifts)].groupby(['호선_노선명', '출발시간대'])['혼잡도'].mean().unstack()[pm_shifts].reindex(ORDERED_LINES)
    pm_bar.plot(kind='bar', ax=axes[1], width=0.78, color=['#1e8449', '#2ecc71', '#e74c3c', '#f39c12', '#3498db'])
    axes[1].axhline(80, color='#1e8449', linestyle='--', linewidth=1.3, label='쾌적 기준선 (80%)')
    axes[1].axhline(115, color='#c0392b', linestyle=':', linewidth=1.3, label='혼잡 경보선 (115%)')
    axes[1].set_title('[PART 2-B] 노선별 퇴근 골든타임 시프트 효과 (18:30 회피)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('노선 / 광역버스')
    axes[1].set_ylabel('평균 혼잡도 (%)')
    axes[1].tick_params(axis='x', rotation=25)
    axes[1].legend(fontsize=8, loc='upper right', ncol=2)
    axes[1].grid(True, axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/fig4_part2_evening_commute.png", dpi=200)
    plt.close()

    # 5. 그림 5: 기상 상태(맑음 vs 비/눈)별 피크타임 혼잡도 급증 비교
    fig, ax = plt.subplots(figsize=(11, 5.2))
    peak_weather = df_clean[(df_clean['평일여부'] == 1) & (df_clean['상하_방향구분'] == '서울방면') & (df_clean['시간_수치'].between(7.5, 8.5))]
    sns.barplot(
        data=peak_weather, x='호선_노선명', y='혼잡도', hue='기상상태',
        order=ORDERED_LINES, palette=['#3498db', '#34495e'], ax=ax, errorbar=None
    )
    ax.axhline(115, color='#e74c3c', linestyle='--', linewidth=1.2, label='혼잡 경보선 (115%)')
    ax.set_title('출근 피크(07:30~08:30) 기상 상태별 혼잡도 급증 비교 (철도 +10%, 광역버스 +14%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('노선 / 광역버스')
    ax.set_ylabel('피크타임 평균 혼잡도 (%)')
    ax.legend(fontsize=9.5)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/fig5_weather_impact_peak.png", dpi=200)
    plt.close()

    # 6. 그림 6: 머신러닝 알고리즘 5종 성능 비교 및 Feature Importance
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    df_res_melt = df_results.melt(id_vars='Model', value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'], var_name='Metric', value_name='Score')
    sns.barplot(data=df_res_melt, x='Model', y='Score', hue='Metric', ax=axes[0], palette='viridis')
    axes[0].set_ylim(0.75, 1.0)
    axes[0].set_title('머신러닝 알고리즘 5종 성능 비교 (쾌적등급 분류)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('평가 점수')
    axes[0].tick_params(axis='x', rotation=20)
    axes[0].legend(loc='lower right', fontsize=8.5)
    axes[0].grid(True, axis='y', linestyle=':', alpha=0.6)

    sns.barplot(data=df_fi, x='Importance', y='Feature', ax=axes[1], palette='magma')
    axes[1].set_title('혼잡도·쾌적등급 예측 주요 영향 요인 (Feature Importance)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('중요도 (Gini / Gain Importance)')
    axes[1].set_ylabel('설명 변수 (Feature)')
    for idx, row in enumerate(df_fi.itertuples()):
        axes[1].text(row.Importance + 0.005, idx, f"{row.Importance:.3f}", va='center', fontsize=9)
    axes[1].grid(True, axis='x', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/fig6_ml_model_comparison_fi.png", dpi=200)
    plt.close()
