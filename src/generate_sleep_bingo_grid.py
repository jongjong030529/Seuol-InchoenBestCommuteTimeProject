# -*- coding: utf-8 -*-
"""
정시(n:00) 및 반시(n:30) 출근 시작 시각 분리형 순수 수면 최적도 5x10 컬러 그리드 생성기
- 혼잡도 완전 제외: 90분 수면주기 경계 기상(수면 관성 최소화) + 입면 지연(+30분)만 반영
- 실행 방법: python src/generate_sleep_bingo_grid.py
"""

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
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


def compute_pure_sleep_cell(w_str: str, lo_str: str, latency_min: int = 30):
    lo_dt = datetime.strptime(lo_str, "%H:%M")
    if lo_dt.hour < 12:
        lo_dt += timedelta(days=1)
    onset_dt = lo_dt + timedelta(minutes=latency_min)
    wake_dt = datetime.strptime(w_str, "%H:%M") + timedelta(days=1)

    sleep_min = (wake_dt - onset_dt).total_seconds() / 60.0
    sleep_hr = sleep_min / 60.0

    rem = sleep_min % 90.0
    dist_cycle = min(rem, 90.0 - rem)
    is_aligned = dist_cycle == 0
    cycles = int(round(sleep_min / 90.0))

    recovery = 100.0 * (1.0 - np.exp(-sleep_hr / 3.35))
    inertia_term = 0.45 * dist_cycle - (5.5 if is_aligned else 0.0)
    if sleep_hr < 3.0:
        inertia_term += (3.0 - sleep_hr) * 20.0

    opt_score = float(np.clip(recovery - inertia_term, 15.0, 99.0))
    return sleep_hr, cycles, is_aligned, int(dist_cycle), round(opt_score, 1)


def _render_single_grid(
    title_badge: str,
    subtitle_text: str,
    commute_list: list[tuple[str, str]],
    out_fig: str,
    out_csv: str,
    accent_color: str,
) -> pd.DataFrame:
    _set_korean_font()
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    lights_out_cols = [
        ("22:00", "22:30 입면"),
        ("22:30", "23:00 입면"),
        ("23:00", "23:30 입면"),
        ("23:30", "00:00 입면"),
        ("00:00", "00:30 입면"),
        ("00:30", "01:00 입면"),
        ("01:00", "01:30 입면"),
        ("01:30", "02:00 입면"),
        ("02:00", "02:30 입면"),
        ("02:30", "03:00 입면"),
    ]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "pure_sleep_grad",
        [
            (0.00, "#922B21"),
            (0.28, "#E74C3C"),
            (0.48, "#F5B041"),
            (0.64, "#F7DC6F"),
            (0.80, "#58D68D"),
            (1.00, "#145A32"),
        ],
    )
    norm = mcolors.Normalize(vmin=29.0, vmax=99.0)

    fig, ax = plt.subplots(figsize=(17.6, 9.6), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.axis("off")
    ax.set_xlim(0, 17.6)
    ax.set_ylim(0, 9.6)

    header_bg = patches.FancyBboxPatch(
        (0.35, 8.48), 16.9, 0.88,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor="#1B2631", edgecolor="none",
    )
    ax.add_patch(header_bg)
    ax.text(0.65, 9.04, title_badge, fontsize=15.5, fontweight="bold", color="#FFFFFF", va="center")
    ax.text(0.65, 8.68, subtitle_text, fontsize=9.8, color="#D5D8DC", va="center")

    row_label_x, row_label_w = 0.35, 2.15
    grid_start_x, cell_w, cell_gap_x = 2.62, 1.40, 0.065
    col_header_y, col_header_h = 7.56, 0.74
    grid_top_y, cell_h, cell_gap_y = 6.56, 0.90, 0.08

    corner_box = patches.FancyBboxPatch(
        (row_label_x, col_header_y), row_label_w, col_header_h,
        boxstyle="round,pad=0.015,rounding_size=0.08",
        facecolor=accent_color, edgecolor="none",
    )
    ax.add_patch(corner_box)
    ax.text(
        row_label_x + row_label_w / 2, col_header_y + col_header_h / 2,
        "출근 시작 시각\n(준비 30분 전 기상)",
        color="#FFFFFF", fontsize=9.8, fontweight="bold", ha="center", va="center",
    )

    for j, (lo_t, onset_t) in enumerate(lights_out_cols):
        cx = grid_start_x + j * (cell_w + cell_gap_x)
        c_bg = "#283747" if lo_t.endswith(":00") else "#34495E"
        cbox = patches.FancyBboxPatch(
            (cx, col_header_y), cell_w, col_header_h,
            boxstyle="round,pad=0.015,rounding_size=0.08",
            facecolor=c_bg, edgecolor="none",
        )
        ax.add_patch(cbox)
        ax.text(cx + cell_w / 2, col_header_y + 0.47, f"{lo_t} 소등", color="#F9E79F", fontsize=10.5, fontweight="bold", ha="center", va="center")
        ax.text(cx + cell_w / 2, col_header_y + 0.19, f"({onset_t})", color="#EAECEE", fontsize=8.4, ha="center", va="center")

    export_rows = []
    for i, (c_t, w_t) in enumerate(commute_list):
        ry = grid_top_y - i * (cell_h + cell_gap_y)
        rbox = patches.FancyBboxPatch(
            (row_label_x, ry), row_label_w, cell_h,
            boxstyle="round,pad=0.015,rounding_size=0.08",
            facecolor="#F4F6F7", edgecolor=accent_color, linewidth=1.6,
        )
        ax.add_patch(rbox)
        ax.text(row_label_x + row_label_w / 2, ry + 0.57, f"{c_t} 출근 시작", color="#1C2833", fontsize=11.2, fontweight="bold", ha="center", va="center")
        ax.text(row_label_x + row_label_w / 2, ry + 0.25, f"기상 {w_t}", color=accent_color, fontsize=9.6, fontweight="bold", ha="center", va="center")

        row_export = {"출근시작시각": c_t, "최적기상시각": w_t}

        for j, (lo_t, _) in enumerate(lights_out_cols):
            cx = grid_start_x + j * (cell_w + cell_gap_x)
            shr, cyc, aligned, d_cyc, opt = compute_pure_sleep_cell(w_t, lo_t)

            rgba = cmap(norm(opt))
            r_c, g_c, b_c, _ = rgba
            lum = 0.299 * r_c + 0.587 * g_c + 0.114 * b_c
            txt_col = "#FFFFFF" if lum < 0.54 else "#17202A"
            sub_col = "#F2F4F4" if lum < 0.54 else "#212F3D"

            edge_col = "#F1C40F" if (aligned and shr >= 4.5) else ("#1C2833" if aligned else "#BDC3C7")
            edge_lw = 2.6 if (aligned and shr >= 4.5) else (1.8 if aligned else 0.7)

            cell = patches.FancyBboxPatch(
                (cx, ry), cell_w, cell_h,
                boxstyle="round,pad=0.012,rounding_size=0.08",
                facecolor=rgba, edgecolor=edge_col, linewidth=edge_lw,
            )
            ax.add_patch(cell)

            if aligned and shr >= 7.5:
                badge = "★ 최상 회복"
            elif aligned and shr == 6.0:
                badge = "★ 평일 최적"
            elif aligned and shr == 4.5:
                badge = "★ 단축 최적"
            elif aligned and shr == 3.0:
                badge = "★ 비상 최소"
            elif shr >= 7.0:
                badge = "양호 (오차30m)"
            elif shr >= 5.0:
                badge = "주의 (중간각성)"
            else:
                badge = "회피 (피로누적)"

            cycle_str = f"{shr:.1f}h · {cyc}주기★" if aligned else f"{shr:.1f}h · 오차{d_cyc}m"
            ax.text(cx + cell_w / 2, ry + 0.64, f"{opt:.1f}점", color=txt_col, fontsize=11.4, fontweight="bold", ha="center", va="center")
            ax.text(cx + cell_w / 2, ry + 0.39, cycle_str, color=sub_col, fontsize=8.4, fontweight="bold", ha="center", va="center")
            ax.text(cx + cell_w / 2, ry + 0.16, badge, color=txt_col, fontsize=7.9, fontweight="bold", ha="center", va="center")

            row_export[f"{lo_t}소등"] = f"{opt:.1f}점 ({cycle_str})"
        export_rows.append(row_export)

    legend_y, legend_h = 0.68, 1.56
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.35, legend_y), 16.9, legend_h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor="#F8F9F9", edgecolor="#D5D8DC", linewidth=1.0,
        )
    )

    cb_x, cb_y, cb_w, cb_h = 0.65, legend_y + 0.72, 5.40, 0.28
    ax.text(cb_x + cb_w / 2, cb_y + 0.48, "순수 수면 최적도 지수 (혼잡도 제외 · 수면 회복 및 주기 일치 기준)", fontsize=9.8, fontweight="bold", color="#1B2631", ha="center", va="center")

    n_steps = 140
    step_w = cb_w / n_steps
    for k in range(n_steps):
        val = 29.0 + (99.0 - 29.0) * (k / (n_steps - 1))
        ax.add_patch(patches.Rectangle((cb_x + k * step_w, cb_y), step_w + 0.002, cb_h, facecolor=cmap(norm(val)), edgecolor="none"))
    ax.add_patch(patches.Rectangle((cb_x, cb_y), cb_w, cb_h, facecolor="none", edgecolor="#2C3E50", linewidth=1.0))

    ticks = [
        (30.0, "30점\n(극단 부족)"),
        (48.0, "48점\n(중간각성)"),
        (65.0, "65점\n(2주기 비상)"),
        (79.4, "79.4점\n(3주기★)"),
        (88.8, "88.8점\n(4주기★)"),
        (98.0, "98점+\n(5~6주기★)"),
    ]
    for tv, tlabel in ticks:
        tx = cb_x + ((tv - 29.0) / (99.0 - 29.0)) * cb_w
        ax.plot([tx, tx], [cb_y - 0.05, cb_y], color="#2C3E50", linewidth=1.0)
        ax.text(tx, cb_y - 0.11, tlabel, fontsize=7.6, color="#212F3D", ha="center", va="top")

    guide_x = 6.45
    ax.text(guide_x, legend_y + 1.24, "핵심 해석 가이드 (혼잡도 제외 · 순수 수면 사이클 기준)", fontsize=10.3, fontweight="bold", color="#1B2631", va="center")
    ax.text(guide_x, legend_y + 0.84, "• 굵은 금색 테두리(★): 90분 수면주기가 정확히 끝나는 시점에 기상해 수면 관성(비몽사몽함)이 최소화된 최적 소등 시각", fontsize=9.1, color="#212F3D", va="center")
    ax.text(guide_x, legend_y + 0.42, "• 역전 구간 주목: 어중간한 5.5시간(오차 30m, 67.1점)보다 90분 주기를 맞춘 4.5시간 3주기(★, 79.4점)가 기상 피로도가 훨씬 낮음", fontsize=9.1, color="#145A32", fontweight="bold", va="center")

    ax.text(0.35, 0.28, "근거 문헌: Hilditch & McHill (2019, Nature and Science of Sleep) | Chang et al. (2015, PNAS)", fontsize=9.0, color="#566573", va="center")

    plt.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.close()

    df_out = pd.DataFrame(export_rows)
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df_out


def generate_split_sleep_grids():
    df_on = _render_single_grid(
        title_badge="[표 1 · 정시 출근(n:00) 기준] 출근 시작 시각 × 전날 소등 시각 5×10 순수 수면 최적도 그리드",
        subtitle_text="※ 혼잡도 제외 · 순수 수면 최적도 | 정시 출근(기상 :30) 특성상 홀수 주기(3·5주기)는 반시(:30) 소등, 짝수 주기(2·4·6주기)는 정시(:00) 소등에서 일치",
        commute_list=[("06:00", "05:30"), ("07:00", "06:30"), ("08:00", "07:30"), ("09:00", "08:30"), ("10:00", "09:30")],
        out_fig="outputs/figures/fig_sleep_grid_on_hour.png",
        out_csv="outputs/sleep_optimal_grid_on_hour.csv",
        accent_color="#1B4F72",
    )
    df_half = _render_single_grid(
        title_badge="[표 2 · 반시 출근(n:30) 기준] 출근 시작 시각 × 전날 소등 시각 5×10 순수 수면 최적도 그리드",
        subtitle_text="※ 혼잡도 제외 · 순수 수면 최적도 | 반시 출근(기상 :00) 특성상 홀수 주기(3·5주기)는 정시(:00) 소등, 짝수 주기(2·4·6주기)는 반시(:30) 소등에서 일치",
        commute_list=[("06:30", "06:00"), ("07:30", "07:00"), ("08:30", "08:00"), ("09:30", "09:00"), ("10:30", "10:00")],
        out_fig="outputs/figures/fig_sleep_grid_half_hour.png",
        out_csv="outputs/sleep_optimal_grid_half_hour.csv",
        accent_color="#1E8449",
    )
    return df_on, df_half


if __name__ == "__main__":
    generate_split_sleep_grids()