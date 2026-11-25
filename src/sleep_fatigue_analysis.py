# -*- coding: utf-8 -*-
"""
현대인 수면 역학 및 정시(n:00)·반시(n:30) 출근 시작 시각 기준 피로도 통계 분석 모듈
- 혼잡도와 분리된 순수 수면 주기(90분) 및 입면 지연(+30분) 역산 체계 반영
- 실행 방법: python src/sleep_fatigue_analysis.py
"""

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def _set_korean_font():
    font_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/malgunbd.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ]
    font_path = next((p for p in font_candidates if os.path.exists(p)), None)
    if font_path:
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False


def build_modern_sleep_dataset(
    output_csv: str = "data/modern_commuter_sleep_dataset.csv",
    n_samples: int = 2400,
    seed: int = 42,
) -> pd.DataFrame:
    """정시(n:00) 및 반시(n:30) 출근 시작 시각을 포함한 수면·피로도 데이터셋 생성"""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    np.random.seed(seed)

    commute_starts = np.random.choice(
        ["06:00", "06:30", "07:00", "07:30", "08:00", "08:30", "09:00", "09:30", "10:00"],
        size=n_samples,
        p=[0.08, 0.12, 0.18, 0.20, 0.16, 0.08, 0.08, 0.06, 0.04],
    )
    schedule_type = np.where(
        pd.Series(commute_starts).str.endswith(":00").values,
        "정시(n:00)",
        "반시(n:30)",
    )

    overtime_days = np.random.choice(
        [0, 1, 2, 3, 4, 5], size=n_samples, p=[0.18, 0.22, 0.25, 0.18, 0.11, 0.06]
    )
    smartphone_min = np.clip(
        np.random.normal(48 + overtime_days * 6, 22, size=n_samples), 0, 150
    ).round(0)
    depression_score = np.clip(
        np.random.normal(6.5 + overtime_days * 1.1 + smartphone_min * 0.03, 3.8, size=n_samples),
        0,
        27,
    ).round(1)

    sleep_latency_min = np.clip(
        12.0
        + 0.14 * smartphone_min
        + 0.65 * depression_score
        + 1.4 * overtime_days
        + np.random.normal(0, 4.5, size=n_samples),
        5,
        75,
    ).round(1)

    actual_sleep_hr = np.clip(
        np.random.normal(6.6 - 0.28 * overtime_days - 0.008 * smartphone_min, 0.95, size=n_samples),
        3.0,
        8.5,
    ).round(2)

    sleep_min = actual_sleep_hr * 60.0
    remainder = sleep_min % 90.0
    dist_from_cycle_min = np.minimum(remainder, 90.0 - remainder)
    cycle_aligned = (dist_from_cycle_min <= 12.0).astype(int)

    risk_score_raw = (
        (overtime_days >= 3).astype(int)
        + (smartphone_min >= 60).astype(int)
        + (depression_score >= 10).astype(int)
    )
    risk_group = np.where(
        risk_score_raw == 0,
        "저위험(안정군)",
        np.where(risk_score_raw == 1, "중위험(단일요인군)", "고위험(야근·스마트폰·우울 복합군)"),
    )

    # 순수 기상 피로도(수면 관성) 점수 (0~100, 낮을수록 쾌적)
    wake_fatigue_score = np.clip(
        82.0
        - 6.2 * actual_sleep_hr
        + 0.36 * dist_from_cycle_min
        - 7.5 * cycle_aligned
        + 0.68 * depression_score
        + 1.35 * overtime_days
        + np.random.normal(0, 4.8, size=n_samples),
        15,
        99,
    ).round(1)

    # 순수 수면 회복 지수 (0~100, 높을수록 쾌적)
    pure_sleep_score = np.clip(100.0 - wake_fatigue_score, 5, 95).round(1)

    df_sleep = pd.DataFrame({
        "출근시작시각": commute_starts,
        "출근구분": schedule_type,
        "주간야근일수": overtime_days,
        "취침전스마트폰_분": smartphone_min,
        "우울스트레스지수": depression_score,
        "현대인위험군": risk_group,
        "입면지연시간_분": sleep_latency_min,
        "실제수면시간_hr": actual_sleep_hr,
        "수면주기오차_분": dist_from_cycle_min.round(1),
        "수면주기일치여부": cycle_aligned,
        "기상피로도점수": wake_fatigue_score,
        "순수수면회복지수": pure_sleep_score,
    })
    df_sleep.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return df_sleep


def _build_schedule_table(commute_times: list[str], group_label: str) -> pd.DataFrame:
    rows = []
    for c_str in commute_times:
        c_dt = datetime.strptime(c_str, "%H:%M")
        wake_dt = c_dt - timedelta(minutes=30)
        row = {
            "출근구분": group_label,
            "출근시작시각": c_str,
            "최적기상시각(준비30분전)": wake_dt.strftime("%H:%M"),
        }
        for cycles, col_name in [
            (2, "3.0시간_2주기(소등/입면)"),
            (3, "4.5시간_3주기(소등/입면)"),
            (4, "6.0시간_4주기(소등/입면)"),
            (5, "7.5시간_5주기(소등/입면)"),
        ]:
            onset_dt = wake_dt - timedelta(minutes=90 * cycles)
            bed_dt = onset_dt - timedelta(minutes=30)
            row[col_name] = f"{bed_dt.strftime('%H:%M')} 소등 ({onset_dt.strftime('%H:%M')} 입면)"
        rows.append(row)
    return pd.DataFrame(rows)


def run_sleep_statistical_analysis(
    df_sleep: pd.DataFrame,
    output_dir: str = "outputs",
    fig_dir: str = "outputs/figures",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """정시·반시 분리 수면 역산표 및 순수 수면 피로도 통계 검정 수행"""
    _set_korean_font()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    aligned_fat = df_sleep[df_sleep["수면주기일치여부"] == 1]["기상피로도점수"]
    misaligned_fat = df_sleep[df_sleep["수면주기일치여부"] == 0]["기상피로도점수"]
    t_stat_cycle, p_val_cycle = stats.ttest_ind(misaligned_fat, aligned_fat, equal_var=False)
    cohen_d_cycle = (misaligned_fat.mean() - aligned_fat.mean()) / np.sqrt(
        (aligned_fat.std() ** 2 + misaligned_fat.std() ** 2) / 2
    )

    short_df = df_sleep[df_sleep["실제수면시간_hr"] <= 4.75]
    short_aligned = short_df[short_df["수면주기일치여부"] == 1]["기상피로도점수"]
    short_misaligned = short_df[short_df["수면주기일치여부"] == 0]["기상피로도점수"]
    t_stat_short, p_val_short = stats.ttest_ind(short_misaligned, short_aligned, equal_var=False)

    g_low = df_sleep[df_sleep["현대인위험군"] == "저위험(안정군)"]["입면지연시간_분"]
    g_mid = df_sleep[df_sleep["현대인위험군"] == "중위험(단일요인군)"]["입면지연시간_분"]
    g_high = df_sleep[df_sleep["현대인위험군"] == "고위험(야근·스마트폰·우울 복합군)"]["입면지연시간_분"]
    f_stat_lat, p_val_lat = stats.f_oneway(g_low, g_mid, g_high)

    on_fat = df_sleep[df_sleep["출근구분"] == "정시(n:00)"]["기상피로도점수"]
    half_fat = df_sleep[df_sleep["출근구분"] == "반시(n:30)"]["기상피로도점수"]
    t_stat_grp, p_val_grp = stats.ttest_ind(on_fat, half_fat, equal_var=False)

    summary_df = pd.DataFrame([
        {
            "검정항목": "전체 표본: 90분 주기 일치 vs 불일치 기상피로도",
            "검정방법": "Welch t-test",
            "비교집단_A": f"주기일치 ({aligned_fat.mean():.2f}점)",
            "비교집단_B": f"주기불일치 ({misaligned_fat.mean():.2f}점)",
            "통계량": f"t = {t_stat_cycle:.2f}",
            "p_value": f"{p_val_cycle:.2e}",
            "효과크기_해석": f"Cohen's d = {cohen_d_cycle:.2f} (매우 큼)",
        },
        {
            "검정항목": "단축수면(≤4.5h): 주기 일치 vs 불일치 기상피로도",
            "검정방법": "Welch t-test",
            "비교집단_A": f"주기일치 ({short_aligned.mean():.2f}점)",
            "비교집단_B": f"주기불일치 ({short_misaligned.mean():.2f}점)",
            "통계량": f"t = {t_stat_short:.2f}",
            "p_value": f"{p_val_short:.2e}",
            "효과크기_해석": f"피로도 -{short_misaligned.mean() - short_aligned.mean():.2f}점 감소",
        },
        {
            "검정항목": "야근·스마트폰·우울 위험군별 입면 지연시간",
            "검정방법": "One-way ANOVA",
            "비교집단_A": f"저위험 ({g_low.mean():.1f}분)",
            "비교집단_B": f"고위험 ({g_high.mean():.1f}분)",
            "통계량": f"F = {f_stat_lat:.2f}",
            "p_value": f"{p_val_lat:.2e}",
            "효과크기_해석": f"고위험군 +{g_high.mean() - g_low.mean():.1f}분 입면 지연",
        },
        {
            "검정항목": "정시(n:00) vs 반시(n:30) 출근군 기상피로도 비교",
            "검정방법": "Welch t-test",
            "비교집단_A": f"정시 n:00 ({on_fat.mean():.2f}점)",
            "비교집단_B": f"반시 n:30 ({half_fat.mean():.2f}점)",
            "통계량": f"t = {t_stat_grp:.2f}",
            "p_value": f"{p_val_grp:.2e}",
            "효과크기_해석": "정시(:30기상)·반시(:00기상) 주기 위상 분리 필요",
        },
    ])
    summary_df.to_csv(os.path.join(output_dir, "sleep_statistical_summary.csv"), index=False, encoding="utf-8-sig")

    df_sched = pd.concat(
        [
            _build_schedule_table(["06:00", "07:00", "08:00", "09:00", "10:00"], "정시(n:00)"),
            _build_schedule_table(["06:30", "07:30", "08:30", "09:30", "10:30"], "반시(n:30)"),
        ],
        ignore_index=True,
    )
    df_sched.to_csv(os.path.join(output_dir, "commute_start_sleep_schedule.csv"), index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")

    ax = axes[0]
    categories = ["전체 표본\n(N=2,400)", "단축수면 통근자\n(≤4.5시간)"]
    x = np.arange(len(categories))
    w = 0.34
    b1 = ax.bar(x - w / 2, [aligned_fat.mean(), short_aligned.mean()], width=w, color="#1E8449", label="90분 주기 일치 기상")
    b2 = ax.bar(x + w / 2, [misaligned_fat.mean(), short_misaligned.mean()], width=w, color="#C0392B", label="주기 중간 각성 기상")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10.5, fontweight="bold")
    ax.set_ylabel("기상 피로도 점수 (낮을수록 쾌적)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_title("A. 90분 수면주기 일치 여부에 따른 기상 피로도", fontsize=11.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=9)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{bar.get_height():.1f}", ha="center", va="bottom", fontweight="bold", fontsize=9.5)

    ax = axes[1]
    groups = ["저위험\n(안정군)", "중위험\n(단일요인군)", "고위험\n(복합군)"]
    means = [g_low.mean(), g_mid.mean(), g_high.mean()]
    stds = [g_low.std(), g_mid.std(), g_high.std()]
    bars_b = ax.bar(groups, means, yerr=stds, capsize=5, width=0.46, color=["#2E86C1", "#F39C12", "#8E44AD"])
    ax.axhline(30, color="#C0392B", linestyle="--", linewidth=1.5, label="현대인 입면 버퍼 기준선 (30분)")
    ax.set_ylabel("평균 입면 지연 시간 (분)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 52)
    ax.set_title("B. 현대인 위험군별 입면 지연 시간 (One-way ANOVA)", fontsize=11.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=9)
    for bar, sd in zip(bars_b, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sd + 1.2, f"{bar.get_height():.1f}분", ha="center", va="bottom", fontweight="bold", fontsize=9.5)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_sleep_statistical_analysis.png"), dpi=200)
    plt.close()

    return summary_df, df_sched


if __name__ == "__main__":
    df_s = build_modern_sleep_dataset("data/modern_commuter_sleep_dataset.csv")
    summary_out, sched_out = run_sleep_statistical_analysis(df_s, "outputs", "outputs/figures")
    print(summary_out.to_string(index=False))
    print(sched_out.to_string(index=False))