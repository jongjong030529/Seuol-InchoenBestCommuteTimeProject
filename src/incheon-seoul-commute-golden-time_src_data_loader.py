# -*- coding: utf-8 -*-
"""
공공데이터 원천 구조(Wide-Format) 로딩 및 표준 스키마 생성 모듈
- 공공데이터포털(data.go.kr) 및 서울 열린데이터광장 개방 데이터 스키마 반영
- 실제 CSV 파일 경로가 주어지면 로드하고, 없을 경우 표준 원천 스키마 데이터를 생성합니다.
"""

import os
import numpy as np
import pandas as pd

TIME_SLOTS = [f"{h:02d}시{m:02d}분" for h in range(5, 24) for m in (0, 30) if not (h == 5 and m == 0)]

SUBWAY_STATIONS = [
    # 1호선 (경인선 축)
    ('1호선', '161', '인천', '인천중부', 0, 0.72),
    ('1호선', '156', '주안', '인천중부', 1, 0.92),
    ('1호선', '152', '부평', '인천중부', 1, 1.15),
    ('1호선', '150', '송내', '인천중부', 0, 1.18),
    ('1호선', '141', '구로', '서울서남', 1, 1.28),
    ('1호선', '140', '신도림', '서울서남', 1, 1.32),
    ('1호선', '133', '서울역', '서울도심', 1, 1.05),
    # 7호선 (부평-가산-강남 축)
    ('7호선', '761', '석남', '인천서북', 1, 0.85),
    ('7호선', '759', '부평구청', '인천중부', 1, 1.04),
    ('7호선', '757', '삼산체육관', '인천중부', 0, 1.08),
    ('7호선', '750', '온수', '서울서남', 1, 1.22),
    ('7호선', '746', '가산디지털단지', '서울서남', 1, 1.35),
    ('7호선', '734', '고속터미널', '서울강남', 1, 1.20),
    ('7호선', '730', '강남구청', '서울강남', 1, 1.10),
    # 공항철도 (청라-검암-계양-서울역 축)
    ('공항철도', 'A071', '청라국제도시', '인천서북', 0, 0.88),
    ('공항철도', 'A07', '검암', '인천서북', 1, 1.12),
    ('공항철도', 'A06', '계양', '인천중부', 1, 1.26),
    ('공항철도', 'A05', '김포공항', '서울서남', 1, 1.30),
    ('공항철도', 'A03', '홍대입구', '서울도심', 1, 1.14),
    ('공항철도', 'A01', '서울역', '서울도심', 1, 1.02),
]

BUS_ROUTES = [
    ('광역버스', 'M6405', '송도센트럴파크(서울방면)', '인천남부', 0, 1.18),
    ('광역버스', 'M6405', '강남역(인천방면)', '서울강남', 1, 1.24),
    ('광역버스', 'M6724', '청라국제업무단지(서울방면)', '인천서북', 0, 1.14),
    ('광역버스', 'M6724', '서울역버스환승센터(인천방면)', '서울도심', 1, 1.16),
    ('광역버스', 'M6117', '검단신도시우미린(서울방면)', '인천서북', 0, 1.20),
    ('광역버스', '1400번', '부평역(서울방면)', '인천중부', 1, 1.10),
    ('광역버스', '9500번', '계양구청(강남방면)', '인천중부', 1, 1.15),
]


def time_to_float(t_str: str) -> float:
    h = int(t_str.split('시')[0])
    m = int(t_str.split('시')[1].replace('분', ''))
    return h + m / 60.0


def _generate_congestion_curve(t_float, mode, direction, day_type, weather, base_factor):
    base = 32.0 if mode != '광역버스' else 38.0
    if direction == '서울방면':
        if day_type.startswith('평일'):
            morning_peak = 88.0 * np.exp(-((t_float - 8.05) ** 2) / (2 * 0.68 ** 2))
            evening_sub = 28.0 * np.exp(-((t_float - 18.3) ** 2) / (2 * 0.90 ** 2))
        else:
            morning_peak = 22.0 * np.exp(-((t_float - 10.5) ** 2) / (2 * 1.4 ** 2))
            evening_sub = 30.0 * np.exp(-((t_float - 16.5) ** 2) / (2 * 1.5 ** 2))
    else:
        if day_type.startswith('평일'):
            morning_peak = 26.0 * np.exp(-((t_float - 8.2) ** 2) / (2 * 0.85 ** 2))
            evening_sub = 90.0 * np.exp(-((t_float - 18.55) ** 2) / (2 * 0.78 ** 2))
        else:
            morning_peak = 20.0 * np.exp(-((t_float - 11.0) ** 2) / (2 * 1.5 ** 2))
            evening_sub = 34.0 * np.exp(-((t_float - 18.0) ** 2) / (2 * 1.5 ** 2))

    val = (base + morning_peak + evening_sub) * base_factor
    if day_type == '평일_월' and t_float < 10.5:
        val *= 1.06
    elif day_type == '평일_금' and t_float >= 17.0:
        val *= 1.08
    elif day_type == '주말':
        val *= 0.62

    if weather == '비_눈':
        val *= (1.14 if mode == '광역버스' else 1.10)

    val += np.random.normal(0, 4.2)
    return round(max(12.0, min(178.0, val)), 1)


def load_or_create_raw_data(raw_csv_path: str = "data/raw_public_transport_wide.csv", seed: int = 42) -> pd.DataFrame:
    """원천 Wide-Format CSV를 로드하거나, 없으면 공공데이터포털 표준 스키마로 생성합니다."""
    if os.path.exists(raw_csv_path):
        return pd.read_csv(raw_csv_path, encoding="utf-8-sig")

    np.random.seed(seed)
    day_types = ['평일_월', '평일_화수목', '평일_금', '주말']
    weather_types = ['맑음', '비_눈']

    raw_rows = []
    row_id = 1
    for mode, line_code, stn_name, region, is_transfer, factor in (SUBWAY_STATIONS + BUS_ROUTES):
        for direction in ['서울방면', '인천방면']:
            for day_type in day_types:
                for weather in weather_types:
                    row = {
                        '연번': row_id,
                        '수단구분': '도시철도' if mode != '광역버스' else '광역버스',
                        '호선_노선명': mode if mode != '광역버스' else line_code,
                        '역_정류장코드': line_code,
                        '출발역_정류장명': stn_name,
                        '인천권역': region,
                        '환승거점여부': is_transfer,
                        '상하_방향구분': direction,
                        '요일구분': day_type,
                        '기상상태': weather
                    }
                    for t_str in TIME_SLOTS:
                        tf = time_to_float(t_str)
                        c_val = _generate_congestion_curve(tf, mode, direction, day_type, weather, factor)
                        if np.random.rand() < 0.012:
                            row[t_str] = np.nan
                        elif np.random.rand() < 0.03:
                            row[t_str] = f"{c_val}%"
                        else:
                            row[t_str] = c_val
                    raw_rows.append(row)
                    row_id += 1

    df_raw = pd.DataFrame(raw_rows)
    os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)
    df_raw.to_csv(raw_csv_path, index=False, encoding="utf-8-sig")
    return df_raw
