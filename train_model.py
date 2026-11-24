"""모델 생성 코드: 토양·기상 조건과 작물 종류로 수확량을 예측하는 회귀 모델

실행 순서
  1. SQLite DB에서 원시 테이블 3개를 SQL JOIN으로 불러오기       (5장)
  2. 전처리: 공백 정리, 누수·중복 정보 열 제거, 0/극단값 처리      (3·4장)
  3. 탐색적 분석: 분포·상관·다중공선성(VIF), 군집·PCA·이상탐지     (4·9장)
  4. 연도 기준 분할(2017~2020년 = 평가), 기준선 계산                (6·7장)
  5. 알고리즘 7종 + 보팅 앙상블 비교 (교차검증 cv=5)               (6·7·8장)
  6. 하이퍼파라미터 튜닝: GridSearchCV / RandomizedSearchCV       (7·8장)
  7. 평가 데이터로 단 1회 최종 평가, 피쳐 중요도
  8. 전체 데이터로 다시 학습해 joblib.dump, 앱에서 쓸 참고값 저장

먼저 python collect_data.py 를 실행해 data/crop.db 를 만들어야 한다.
"""

import json
import sqlite3
import time
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")  # 그림을 화면에 띄우지 않고 파일로 저장
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy.stats import randint  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import (HistGradientBoostingRegressor, IsolationForest,  # noqa: E402
                              RandomForestRegressor, VotingRegressor)
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import Lasso, LinearRegression, Ridge  # noqa: E402
from sklearn.metrics import (mean_absolute_error, mean_squared_error,  # noqa: E402
                             r2_score, silhouette_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,  # noqa: E402
                                     cross_validate)
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

from common import (CATEGORICAL_FEATURES, CROP_NAMES_KO, DB_PATH,  # noqa: E402
                    EXCLUDED_CROPS, FEATURE_LABELS, FEATURES, FIGURE_DIR,
                    MIN_ROWS_PER_CROP, MODEL_PATH, NUMERIC_FEATURES, RANDOM_STATE,
                    REFERENCE_PATH, SOIL_FEATURES, TARGET, TEST_START_YEAR,
                    WEATHER_FEATURES)


def set_korean_font():
    """강의 저장소는 D2Coding을 사용. 없으면 OS 기본 한글 글꼴을 순서대로 찾는다."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["D2Coding", "Malgun Gothic", "AppleGothic", "NanumGothic",
                 "Noto Sans CJK KR", "Noto Sans CJK JP"]:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 1. 데이터 불러오기: 세 테이블을 SQL JOIN 으로 합친다
#    작물 통계(주·연도·작물) ← 토양(주) ← 기상(주·연도)
# ---------------------------------------------------------------------------
QUERY = """
SELECT TRIM(c.crop)   AS crop,
       TRIM(c.season) AS season,
       TRIM(c.state)  AS state,
       c.year, c.area, c.production, c.fertilizer, c.pesticide, c.yield,
       s.N AS n, s.P AS p, s.K AS k, s.pH AS ph,
       w.avg_temp_c, w.total_rainfall_mm, w.avg_humidity_percent
FROM crop_yield AS c
JOIN state_soil    AS s ON TRIM(c.state) = s.state
JOIN state_weather AS w ON TRIM(c.state) = w.state AND c.year = w.year
"""


def load_data():
    if not DB_PATH.is_file():
        raise SystemExit(f"DB 파일이 없습니다: {DB_PATH}\n먼저 python collect_data.py 를 실행해 주세요.")
    with sqlite3.connect(DB_PATH) as conn:
        raw_rows = pd.read_sql("SELECT COUNT(*) AS n FROM crop_yield", conn)["n"][0]
        df = pd.read_sql(QUERY, conn)
    print(f"[1] 원시 작물 통계 {raw_rows:,}행 → 토양·기상과 결합 {len(df):,}행 "
          f"(결합 실패 {raw_rows - len(df)}행)")
    return df


# ---------------------------------------------------------------------------
# 2. 전처리
# ---------------------------------------------------------------------------
def preprocess(df):
    df = df.copy()
    log = []

    # (1) 중복 확인: 같은 (작물, 연도, 재배철, 주) 조합이 두 번 나오면 안 된다
    dup = df.duplicated(["crop", "year", "season", "state"]).sum()
    log.append(f"중복 키 {dup}건")

    # (2) 정답 누수·정보가 겹치는 열 제거
    #   - production, area : yield ≈ production / area 로 계산된 값이라 넣으면 정답을 알려주는 셈
    ratio = (df["yield"] / (df["production"] / df["area"])).replace([np.inf, -np.inf], np.nan)
    log.append(f"yield / (production/area) 중앙값 {ratio.median():.3f} → production, area 제거")
    #   - fertilizer, pesticide : 면적 × '그해 전국 평균 사용량'이라, 면적으로 나누면 연도마다 값이 하나뿐
    fert_rate_std = (df["fertilizer"] / df["area"]).groupby(df["year"]).std().max()
    log.append(f"연도 안에서 (비료/면적)의 표준편차 최대 {fert_rate_std:.2e} → 연도와 같은 정보라 제거")
    #   - state : 주마다 토양값이 하나로 정해져 있어 토양 4개 피쳐와 정보가 겹침 (분석용으로만 보관)
    df = df.drop(columns=["production", "area", "fertilizer", "pesticide"])

    # (3) 추천할 수 없는 항목 제거: 집계 항목(Other ~), 단위가 다른 코코넛
    before = len(df)
    df = df[~df["crop"].isin(EXCLUDED_CROPS)]
    log.append(f"집계·단위 상이 항목 제거 {before - len(df)}행")

    # (4) 수확량 0 제거: 재배 실패 또는 미집계가 0으로 기록된 행 (log 변환도 불가능)
    before = len(df)
    df = df[df[TARGET] > 0]
    log.append(f"수확량 0 제거 {before - len(df)}행")

    # (5) 작물별 극단값 제거 — 수정 z-점수(modified z-score, Iglewicz & Hoaglin 1993)
    #     z = 0.6745 × (x − 중앙값) / MAD,  |z| > 3.5 이면 이상치로 본다.
    #     작물마다 단위·규모가 달라 전체 기준이 아닌 '작물별' 기준을 쓰고, 분포가 오른쪽으로
    #     길어서 log(수확량)에 적용한다. 실제로 걸러지는 값은 양파 381톤/ha, 옥수수 990톤/ha,
    #     델리 감자 311톤/ha처럼 농학적으로 불가능한 입력 오류들이다.
    #     (처음에는 기준을 5로 두었는데 이런 값이 남아 평가 R²를 크게 떨어뜨려서 3.5로 바꿨다)
    logy = np.log(df[TARGET])
    med = logy.groupby(df["crop"]).transform("median")
    mad = (logy - med).abs().groupby(df["crop"]).transform("median")
    modified_z = 0.6745 * (logy - med) / mad
    before = len(df)
    df = df[modified_z.abs() <= 3.5]
    log.append(f"작물별 극단값(수정 z-점수 |z| > 3.5) 제거 {before - len(df)}행")

    # (6) 관측치가 너무 적은 작물 제거 (학습·평가가 불안정)
    counts = df["crop"].value_counts()
    rare = counts[counts < MIN_ROWS_PER_CROP].index
    before = len(df)
    df = df[~df["crop"].isin(rare)]
    log.append(f"관측치 {MIN_ROWS_PER_CROP}개 미만 작물 {len(rare)}종 제거 {before - len(df)}행 "
               f"({', '.join(rare)})")

    df = df.sort_values(["year", "state", "crop", "season"]).reset_index(drop=True)
    for line in log:
        print("[2]", line)
    print(f"[2] 전처리 후 {len(df):,}행, 작물 {df['crop'].nunique()}종, "
          f"재배철 {df['season'].nunique()}종, 주 {df['state'].nunique()}개")
    return df


# ---------------------------------------------------------------------------
# 3. 탐색적 분석
# ---------------------------------------------------------------------------
def vif_table(X):
    """다중공선성 지표 VIF = 1 / (1 - R²_j). 10 이상이면 정보가 크게 겹친다고 본다."""
    rows = []
    for col in X.columns:
        others = X.drop(columns=col)
        r2 = LinearRegression().fit(others, X[col]).score(others, X[col])
        rows.append({"피쳐": col, "VIF": 1 / (1 - r2)})
    return pd.DataFrame(rows)


def explore(df):
    FIGURE_DIR.mkdir(exist_ok=True)

    # (a) 수확량 분포: 오른쪽 꼬리가 길어 log 변환을 고려
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df[TARGET], bins=60, ax=axes[0])
    axes[0].set_title("수확량 분포 (원래 값)")
    sns.histplot(np.log1p(df[TARGET]), bins=60, ax=axes[1], color="#55A868")
    axes[1].set_title("수확량 분포 (log1p 변환)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "eda_target_distribution.png", dpi=120)
    plt.close(fig)

    # (b) 작물별 수확량 상자그림 (관측치 많은 15종, 로그 축)
    top = df["crop"].value_counts().index[:15]
    sub = df[df["crop"].isin(top)].assign(작물=lambda d: d["crop"].map(CROP_NAMES_KO))
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=sub, x="작물", y=TARGET, ax=ax)
    ax.set_yscale("log")
    ax.set_title("작물별 수확량 분포 (관측치 상위 15종, 로그 축)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "eda_crop_boxplot.png", dpi=120)
    plt.close(fig)

    # (c) 수치 피쳐 상관계수 + VIF (서로 겹치는 정보가 없는지 확인)
    corr = df[NUMERIC_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr.rename(index=FEATURE_LABELS, columns=FEATURE_LABELS),
                annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("입력 피쳐 간 상관계수")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "eda_correlation.png", dpi=120)
    plt.close(fig)
    vif = vif_table(df[NUMERIC_FEATURES])
    print("[3] VIF:", ", ".join(f"{r.피쳐} {r.VIF:.1f}" for r in vif.itertuples()))

    # (d) 9장: 주(州)별 토양·기후 프로필 군집 (K-평균 + PCA 2차원 시각화)
    #     토양값은 주마다 하나이므로 주 단위(30개)로 묶어서 본다.
    states = df.groupby("state")[NUMERIC_FEATURES].mean()
    Z = StandardScaler().fit_transform(states)
    scores = {k: silhouette_score(Z, KMeans(n_clusters=k, n_init=10,
                                            random_state=RANDOM_STATE).fit_predict(Z))
              for k in range(2, 7)}
    best_k = max(scores, key=scores.get)
    labels = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit_predict(Z)
    pca = PCA(n_components=2).fit(Z)
    P = pca.transform(Z)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(P[:, 0], P[:, 1], c=labels, cmap="tab10", s=60)
    for (x, y), name in zip(P, states.index):
        ax.annotate(name, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_title(f"주별 토양·기후 프로필 (PCA 2차원, K-평균 k={best_k})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax.legend(*sc.legend_elements(), title="군집", loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "eda_soil_clusters.png", dpi=120)
    plt.close(fig)
    print(f"[3] 주별 토양·기후 군집: 실루엣 점수 최고 k={best_k} "
          f"({scores[best_k]:.2f}), PCA 2개 성분 설명력 {pca.explained_variance_ratio_.sum():.0%}"
          " → 탐색용 해석에만 사용, 모델 입력에는 넣지 않음")

    # (e) 9장: IsolationForest 로 이상치 '검사' (전처리에서 이미 작물별 극단값을 제거했으므로
    #     남은 의심 행이 특정 작물·연도에 몰려 있는지 확인만 한다)
    X_iso = pd.concat([np.log1p(df[[TARGET]]), df[NUMERIC_FEATURES]], axis=1)
    flag = IsolationForest(contamination=0.02, random_state=RANDOM_STATE).fit_predict(X_iso) == -1
    top_crops = df.loc[flag, "crop"].value_counts().head(3).to_dict()
    print(f"[3] IsolationForest 이상치 의심 {flag.sum()}행 (상위 작물 {top_crops}) "
          "→ 기상이 극단적인 해(가뭄·폭우)의 실제 관측이므로 삭제하지 않음")

    return vif


# ---------------------------------------------------------------------------
# 4~5. 모델 정의와 비교
# ---------------------------------------------------------------------------
def make_pipeline(model, scale):
    """수치 피쳐는 (필요한 모델만) 표준화, 범주 피쳐는 원-핫 인코딩.
    전처리를 파이프라인 안에 넣어 교차검증 시 데이터 누수를 막는다 (8장)."""
    numeric = StandardScaler() if scale else "passthrough"
    prep = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])
    pipe = Pipeline([("prep", prep), ("model", model)])
    # 수확량은 오른쪽 꼬리가 긴 분포 → log1p 로 학습하고 예측은 원래 단위로 되돌린다
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def candidates():
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    hgb = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    return {
        "선형 회귀": make_pipeline(LinearRegression(), scale=True),
        "릿지": make_pipeline(Ridge(alpha=1.0), scale=True),
        "라쏘": make_pipeline(Lasso(alpha=0.001, max_iter=10000), scale=True),
        "k-최근접 이웃": make_pipeline(KNeighborsRegressor(n_neighbors=10), scale=True),
        "결정 트리": make_pipeline(DecisionTreeRegressor(max_depth=12,
                                                       random_state=RANDOM_STATE), scale=False),
        "랜덤 포레스트": make_pipeline(rf, scale=False),
        "그레이디언트 부스팅": make_pipeline(hgb, scale=False),
        "보팅(RF+GB)": make_pipeline(VotingRegressor([("rf", clone(rf)), ("gb", clone(hgb))]),
                                   scale=False),
    }


def evaluate(y_true, y_pred):
    return {"R2": r2_score(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2_log": r2_score(np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None)))}


def main():
    set_korean_font()
    df = preprocess(load_data())
    vif = explore(df)

    # 4. 연도 기준 분할: 과거(1997~2016)로 학습하고 미래(2017~2020)를 예측해 평가
    #    (무작위 분할은 같은 주·작물의 인접 연도가 학습·평가에 섞여 성능이 부풀려진다)
    train = df[df["year"] < TEST_START_YEAR]
    test = df[df["year"] >= TEST_START_YEAR]
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    print(f"[4] 학습 {len(train):,}행 (1997~{TEST_START_YEAR - 1}) / "
          f"평가 {len(test):,}행 ({TEST_START_YEAR}~2020)")

    # 기준선: 토양·기상을 전혀 보지 않고 '그 작물·재배철의 과거 중앙값'으로 찍는 방법
    med = train.groupby(["crop", "season"])[TARGET].median()
    crop_med = train.groupby("crop")[TARGET].median()
    base_pred = [med.get((c, s), crop_med[c]) for c, s in zip(test["crop"], test["season"])]
    baseline = evaluate(y_test, np.array(base_pred))
    print(f"[4] 기준선(작물·재배철 중앙값) 평가 R² {baseline['R2']:.3f}")

    # 5. 알고리즘 비교: 학습 데이터에서 5겹 교차검증 (연도순으로 정렬되어 있어 연도 블록 단위로 나뉨)
    rows = []
    for name, model in candidates().items():
        start = time.time()
        s = cross_validate(model, X_train, y_train, cv=5,
                           scoring=("r2", "neg_root_mean_squared_error"))
        rows.append({"알고리즘": name, "CV_R2": s["test_r2"].mean(),
                     "CV_R2_std": s["test_r2"].std(),
                     "CV_RMSE": -s["test_neg_root_mean_squared_error"].mean()})
        print(f"    {name:<12} CV R² {rows[-1]['CV_R2']:.3f} ± {rows[-1]['CV_R2_std']:.3f}, "
              f"RMSE {rows[-1]['CV_RMSE']:.2f} ({time.time() - start:.0f}초)")
    comparison = pd.DataFrame(rows).sort_values("CV_R2", ascending=False)
    print(f"[5] 교차검증 1위: {comparison.iloc[0]['알고리즘']}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=comparison, y="알고리즘", x="CV_R2", color="#4C72B0", ax=ax)
    ax.axvline(0.8, color="red", linestyle="--", label="목표 R² 0.8")
    ax.set_xlim(0, 1)
    ax.set_title("알고리즘별 5겹 교차검증 R² (학습 데이터)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_comparison.png", dpi=120)
    plt.close(fig)

    # 6. 하이퍼파라미터 튜닝: 상위 앙상블 두 종을 각각 다른 방법으로 탐색 (7장)
    #    - 그레이디언트 부스팅: GridSearchCV (격자 전부 시도)
    #    - 랜덤 포레스트     : RandomizedSearchCV (분포에서 n_iter개 무작위 시도)
    m = "regressor__model__"
    searches = {
        "그레이디언트 부스팅": GridSearchCV(
            candidates()["그레이디언트 부스팅"],
            {m + "learning_rate": [0.05, 0.1],
             m + "max_iter": [200, 400],
             m + "max_leaf_nodes": [31, 63],
             m + "min_samples_leaf": [10, 30]},
            cv=5, scoring="r2", n_jobs=-1),
        "랜덤 포레스트": RandomizedSearchCV(
            candidates()["랜덤 포레스트"],
            {m + "n_estimators": randint(100, 250),
             m + "max_depth": [None, 15, 25],
             m + "min_samples_leaf": randint(1, 6),
             m + "max_features": [0.5, 0.8]},
            n_iter=6, cv=5, scoring="r2", random_state=RANDOM_STATE, n_jobs=1),
    }
    tuning = []
    best_name, best_search = None, None
    for name, search in searches.items():
        start = time.time()
        search.fit(X_train, y_train)
        params = {k.replace(m, ""): v for k, v in search.best_params_.items()}
        before = comparison.set_index("알고리즘").loc[name, "CV_R2"]
        tuning.append({"알고리즘": name, "방법": type(search).__name__,
                       "튜닝 전 CV_R2": before, "튜닝 후 CV_R2": search.best_score_,
                       "최적 하이퍼파라미터": params})
        print(f"[6] {name} ({type(search).__name__}, {time.time() - start:.0f}초): "
              f"CV R² {before:.3f} → {search.best_score_:.3f}  {params}")
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_name, best_search = name, search
    best_params = next(t["최적 하이퍼파라미터"] for t in tuning if t["알고리즘"] == best_name)
    print(f"    최종 선택: {best_name}")

    # 7. 평가 데이터로 단 1회 최종 평가
    final = best_search.best_estimator_
    train_metrics = evaluate(y_train, final.predict(X_train))
    pred = final.predict(X_test)
    test_metrics = evaluate(y_test, pred)
    print(f"[7] 최종 평가 R² {test_metrics['R2']:.3f} (학습 R² {train_metrics['R2']:.3f}), "
          f"RMSE {test_metrics['RMSE']:.2f}, MAE {test_metrics['MAE']:.2f}, "
          f"log 척도 R² {test_metrics['R2_log']:.3f}")
    print(f"    기준선 대비: R² {baseline['R2']:.3f} → {test_metrics['R2']:.3f}")

    # 작물마다 수확량 규모가 달라 오차도 작물별로 따로 계산해 둔다 (앱에서 예상 오차로 안내)
    crop_mae = (pd.DataFrame({"crop": test["crop"], "err": (y_test - pred).abs()})
                .groupby("crop")["err"].median())

    imp = permutation_importance(final, X_test, y_test, n_repeats=5, scoring="r2",
                                 random_state=RANDOM_STATE, n_jobs=-1)
    importance = (pd.Series(imp.importances_mean, index=FEATURES)
                  .sort_values(ascending=False))
    print("    순열 중요도(R² 감소량):",
          ", ".join(f"{k} {v:.3f}" for k, v in importance.items()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].scatter(y_test, pred, s=6, alpha=0.4)
    lim = [y_test.min() * 0.8, y_test.max() * 1.2]
    axes[0].plot(lim, lim, "r--")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("실제 수확량")
    axes[0].set_ylabel("예측 수확량")
    axes[0].set_title(f"평가 데이터(2017~2020): 실제 vs 예측, R² {test_metrics['R2']:.3f}")
    labels = {**FEATURE_LABELS, "crop": "작물 종류", "season": "재배철"}
    importance.rename(labels).iloc[::-1].plot.barh(ax=axes[1], color="#4C72B0")
    axes[1].set_title("순열 중요도 (섞었을 때 R² 감소량)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "final_evaluation.png", dpi=120)
    plt.close(fig)

    # 8. 서비스용 모델: 평가가 끝났으므로 2020년까지 전체 데이터로 다시 학습해 저장
    service_model = clone(final).fit(df[FEATURES], df[TARGET])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(service_model, MODEL_PATH, compress=3)  # 압축 저장 (GitHub 업로드 용량 고려)

    # 앱에서 쓰는 참고값: 작물·재배철별 관측 수와 중앙값(=보통 수확량), 입력 허용 범위, 성능
    cs = df.groupby(["crop", "season"])[TARGET].agg(["count", "median"]).reset_index()
    reference = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "algorithm": best_name,
        "best_params": {k: (v if isinstance(v, (int, float, str)) or v is None else str(v))
                        for k, v in best_params.items()},
        "test_period": f"{TEST_START_YEAR}-2020",
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "baseline_metrics": baseline,
        "cv_comparison": comparison.to_dict(orient="records"),
        "tuning": [{**t, "최적 하이퍼파라미터": {k: str(v) for k, v in
                                          t["최적 하이퍼파라미터"].items()}} for t in tuning],
        "permutation_importance": importance.to_dict(),
        "crop_test_median_abs_error": crop_mae.round(4).to_dict(),
        "vif": dict(zip(vif["피쳐"], vif["VIF"])),
        "input_ranges": {c: [float(df[c].min()), float(df[c].max())] for c in NUMERIC_FEATURES},
        "soil_features": SOIL_FEATURES,
        "weather_features": WEATHER_FEATURES,
        "crop_season_stats": [
            {"crop": r.crop, "season": r.season, "count": int(r["count"]),
             "median_yield": float(r["median"])} for _, r in cs.iterrows()],
        "rows_used": len(df),
    }
    REFERENCE_PATH.write_text(json.dumps(reference, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"[8] 모델 저장: {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"    참고값 저장: {REFERENCE_PATH}")
    print("    정확도 목표(평가 R² ≥ 0.80):", "달성" if test_metrics["R2"] >= 0.8 else "미달")


if __name__ == "__main__":
    main()
