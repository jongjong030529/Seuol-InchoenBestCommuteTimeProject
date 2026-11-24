# -*- coding: utf-8 -*-
"""
머신러닝 모델 학습, 교차검증 및 성능 평가 모듈 (scikit-learn)
- 분류 모델 5종: HistGradientBoosting, Gradient Boosting, Decision Tree, Random Forest, Logistic Regression
- 회귀 모델 1종: Random Forest Regressor (연속형 혼잡도 % 예측)
- 주요 영향 요인(Feature Importance) 산출 및 CSV 저장
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestRegressor
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score
)

CAT_FEATURES = ['수단구분', '호선_노선명', '출발역_정류장명', '인천권역', '상하_방향구분', '요일구분', '통근구간구분']
NUM_FEATURES = ['시간_수치', '환승거점여부', '평일여부', '강수여부']


def train_and_evaluate(df_clean: pd.DataFrame, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)

    X = df_clean[CAT_FEATURES + NUM_FEATURES]
    y_cls = df_clean['쾌적등급']
    y_reg = df_clean['혼잡도']

    X_train, X_test, y_train, y_test, yr_train, yr_test = train_test_split(
        X, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
    )

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_FEATURES),
        ('num', StandardScaler(), NUM_FEATURES)
    ])

    models = {
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=6, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=14, random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

    results = []
    trained_pipelines = {}

    for name, clf in models.items():
        pipe = Pipeline([('prep', preprocessor), ('model', clf)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        results.append({
            'Model': name,
            'Accuracy': round(accuracy_score(y_test, preds), 4),
            'Precision': round(precision_score(y_test, preds, average='weighted'), 4),
            'Recall': round(recall_score(y_test, preds, average='weighted'), 4),
            'F1-Score': round(f1_score(y_test, preds, average='weighted'), 4)
        })
        trained_pipelines[name] = pipe

    df_results = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
    df_results.to_csv(f"{output_dir}/model_comparison_results.csv", index=False, encoding="utf-8-sig")

    # 회귀 모델 (혼잡도 % 수치 직접 예측)
    reg_pipe = Pipeline([
        ('prep', preprocessor),
        ('reg', RandomForestRegressor(n_estimators=150, max_depth=14, random_state=42, n_jobs=-1))
    ])
    reg_pipe.fit(X_train, yr_train)
    yr_pred = reg_pipe.predict(X_test)
    reg_metrics = {
        "RMSE": round(float(np.sqrt(mean_squared_error(yr_test, yr_pred))), 2),
        "R2": round(float(r2_score(yr_test, yr_pred)), 4)
    }

    # Feature Importance 산출 (Gradient Boosting 기준 원본 변수 단위 합산)
    gb_pipe = trained_pipelines['Gradient Boosting']
    ohe_names = gb_pipe.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(CAT_FEATURES)
    all_feature_names = list(ohe_names) + NUM_FEATURES
    importances = gb_pipe.named_steps['model'].feature_importances_

    grouped_fi = {}
    for fname, imp in zip(all_feature_names, importances):
        matched = False
        for cf in CAT_FEATURES:
            if fname.startswith(cf + '_'):
                grouped_fi[cf] = grouped_fi.get(cf, 0.0) + imp
                matched = True
                break
        if not matched:
            grouped_fi[fname] = grouped_fi.get(fname, 0.0) + imp

    df_fi = pd.DataFrame(list(grouped_fi.items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    df_fi['Importance_Pct'] = (df_fi['Importance'] * 100).round(2)
    df_fi.to_csv(f"{output_dir}/feature_importance.csv", index=False, encoding="utf-8-sig")

    return df_results, df_fi, reg_metrics, trained_pipelines['HistGradientBoosting'], reg_pipe
