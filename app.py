"""토양 조건별 작물 추천 CLI (응용 SW)

train_model.py 가 joblib 으로 저장한 회귀 모델을 불러와, 사용자가 입력한 토양·기상 조건으로
작물별 수확량을 예측하고 다음 세 가지 기능을 제공한다.

  1. 작물 수확량 예측       : 고른 작물 하나의 예상 수확량과 '보통 해' 대비 수준
  2. 토양 맞춤 작물 추천     : 그 재배철에 기를 수 있는 모든 작물을 예측해 적합도 순위 TOP 5
  3. 토양 개선 시뮬레이션    : N·P·K·pH 를 조정했을 때 수확량이 가장 많이 오르는 방향 제시

참고: 입력 검증 함수와 모델 파일 확인 방식은 강의 예제
      cgsong0730/ds_and_ai assignment/ex_auto_logistic_app.py 의 구조를 참고했다.
"""

import json
import math
import unicodedata

import joblib
import pandas as pd

from common import (FEATURE_LABELS, MODEL_PATH, REFERENCE_PATH, SEASON_NAMES_KO,
                    SOIL_FEATURES, WEATHER_FEATURES, crop_label, yield_unit)

MIN_RECORDS_FOR_SEASON = 30   # 그 재배철 재배 기록이 이보다 적은 작물은 추천 후보에서 제외
MENU = {}                     # 번호: (메뉴 이름, 실행 함수) — 기능마다 아래에서 등록


# ===========================================================================
# 공통: 모델·참고값 불러오기, 입력 받기
# ===========================================================================
def load_resources():
    if not MODEL_PATH.is_file() or not REFERENCE_PATH.is_file():
        raise SystemExit(
            f"모델 파일이 없습니다: {MODEL_PATH}\n"
            "먼저 python collect_data.py → python train_model.py 순서로 실행해 주세요.")
    model = joblib.load(MODEL_PATH)                     # 저장된 모델 불러오기
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    stats = pd.DataFrame(reference["crop_season_stats"])
    return model, reference, stats


def read_number(prompt, minimum, maximum):
    """범위 안의 숫자를 받을 때까지 다시 묻는다 (학습 데이터 범위 밖은 예측을 믿기 어렵기 때문)"""
    while True:
        text = input(prompt).strip()
        try:
            value = float(text)
        except ValueError:
            print("  숫자를 입력해 주세요.")
            continue
        if not math.isfinite(value) or not minimum <= value <= maximum:
            print(f"  {minimum:g}부터 {maximum:g}까지의 값을 입력해 주세요. (학습 데이터 범위)")
        else:
            return value


def choose(prompt, options):
    """번호 목록에서 하나를 고른다. options: [(값, 화면에 보일 이름), ...]"""
    for i, (_, label) in enumerate(options, start=1):
        print(f"  {i:>2}. {label}")
    while True:
        text = input(prompt).strip()
        if text.isdigit() and 1 <= int(text) <= len(options):
            return options[int(text) - 1][0]
        print(f"  1부터 {len(options)}까지의 번호를 입력해 주세요.")


def ask_yes_no(prompt, default=True):
    text = input(prompt).strip().lower()
    if not text:
        return default
    return text in ("y", "yes", "예", "네", "ㅇ")


class ConditionStore:
    """토양·기상 조건은 한 번 입력하면 다른 기능에서도 다시 쓸 수 있게 기억한다."""

    def __init__(self, ranges):
        self.ranges = ranges
        self.values = None

    def get(self):
        if self.values and ask_yes_no("저장된 토양·기상 조건을 그대로 쓸까요? (Y/n): "):
            return dict(self.values)
        r = self.ranges
        print("\n[토양 조건] 토양 검정 결과를 입력해 주세요. (괄호 안은 학습 데이터 범위)")
        values = {}
        for col in SOIL_FEATURES:
            lo, hi = r[col]
            values[col] = read_number(f"  {FEATURE_LABELS[col]} ({lo:g}~{hi:g}): ", lo, hi)
        print("[기상 조건] 재배 지역의 연간 기후를 입력해 주세요.")
        for col in WEATHER_FEATURES:
            lo, hi = r[col]
            values[col] = read_number(f"  {FEATURE_LABELS[col]} ({lo:g}~{hi:g}): ", lo, hi)
        self.values = values
        return dict(values)


def predict(model, conditions, crops, season):
    """같은 토양·기상 조건으로 여러 작물의 수확량을 한 번에 예측한다."""
    rows = pd.DataFrame([{**conditions, "crop": c, "season": season} for c in crops])
    return model.predict(rows)


def choose_season(stats, crop=None):
    sub = stats if crop is None else stats[stats["crop"] == crop]
    seasons = (sub.groupby("season")["count"].sum().sort_values(ascending=False).index)
    return choose("재배철 번호: ", [(s, SEASON_NAMES_KO.get(s, s)) for s in seasons])


def choose_crop(stats, season=None):
    sub = stats if season is None else stats[stats["season"] == season]
    crops = sorted(sub["crop"].unique(), key=lambda c: crop_label(c))
    return choose("작물 번호: ", [(c, crop_label(c)) for c in crops])


def typical_yield(stats, crop, season):
    row = stats[(stats["crop"] == crop) & (stats["season"] == season)]
    return float(row["median_yield"].iloc[0]), int(row["count"].iloc[0])


# ===========================================================================
# 기능 1. 작물 수확량 예측
# ===========================================================================
def feature_predict_yield(model, reference, stats, store):
    print("\n━━ 1. 작물 수확량 예측 ━━")
    print("어떤 작물을 기를지 골라 주세요.")
    crop = choose_crop(stats)
    print("재배철을 골라 주세요. (이 작물이 실제로 재배된 철만 표시)")
    season = choose_season(stats, crop)
    conditions = store.get()

    pred = predict(model, conditions, [crop], season)[0]
    typical, count = typical_yield(stats, crop, season)
    crop_err = reference.get("crop_test_median_abs_error", {}).get(crop)
    unit = yield_unit(crop)
    ratio = pred / typical

    print(f"\n▶ {crop_label(crop)} · {SEASON_NAMES_KO.get(season, season)}")
    print(f"  예상 수확량 : {pred:.2f} {unit}")
    print(f"  보통 수확량 : {typical:.2f} {unit} (과거 {count}건의 중앙값)")
    level = ("매우 좋음" if ratio >= 1.3 else "좋음" if ratio >= 1.1 else
             "보통" if ratio >= 0.9 else "낮음" if ratio >= 0.7 else "매우 낮음")
    print(f"  보통 대비   : {ratio:.0%} → {level}")
    if crop_err is not None:
        print(f"  예상 오차   : ±{crop_err:.2f} {unit} 안팎 (2017~2020년 평가에서 이 작물의 오차 중앙값)")
    if area_text := input("재배 면적(ha)을 입력하면 총 예상 생산량을 계산합니다 (Enter=건너뛰기): ").strip():
        try:
            area = float(area_text)
            if area > 0 and math.isfinite(area):
                print(f"  총 예상 생산량: {pred * area:,.1f} {unit.split('/')[0]} ({area:g} ha)")
            else:
                print("  0보다 큰 면적을 입력해야 계산할 수 있습니다.")
        except ValueError:
            print("  숫자가 아니어서 생산량 계산은 건너뜁니다.")


MENU["1"] = ("작물 수확량 예측", feature_predict_yield)


# ===========================================================================
# 기능 2. 토양 맞춤 작물 추천 (TOP 5)
#   작물마다 수확량 단위·규모가 달라(사탕수수 수십 톤 vs 참깨 1톤 미만) 예측값을 그대로
#   비교하면 늘 사탕수수가 1위가 된다. 그래서 '예측 수확량 ÷ 그 작물의 보통 수확량'
#   = 적합도 지수로 순위를 매긴다. 1.2 이면 그 작물의 보통 해보다 20% 더 잘 자란다는 뜻.
# ===========================================================================
def pad(text, width, align="left"):
    """한글은 화면에서 두 칸을 차지하므로, 실제 표시 폭 기준으로 공백을 채워 표 칸을 맞춘다."""
    shown = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))
    space = " " * max(width - shown, 0)
    return f"{space}{text}" if align == "right" else f"{text}{space}"


def feature_recommend_crops(model, reference, stats, store):
    print("\n━━ 2. 토양 맞춤 작물 추천 ━━")
    print("언제 재배할 계획인가요?")
    season = choose_season(stats)
    conditions = store.get()

    pool = stats[(stats["season"] == season) & (stats["count"] >= MIN_RECORDS_FOR_SEASON)]
    preds = predict(model, conditions, pool["crop"].tolist(), season)
    ranking = pool.assign(pred=preds, index=preds / pool["median_yield"].to_numpy())
    ranking = ranking.sort_values("index", ascending=False).reset_index(drop=True)

    top_n = min(5, len(ranking))
    print(f"\n▶ {SEASON_NAMES_KO.get(season, season)} 후보 {len(ranking)}종 중 적합도 상위 {top_n}종")
    print("  " + pad("순위", 6) + pad("작물", 34) + pad("적합도", 8, "right")
          + pad("예상 수확량", 18, "right") + pad("보통 수확량", 18, "right"))
    for i, row in ranking.head(top_n).iterrows():
        unit = yield_unit(row["crop"])
        print("  " + pad(i + 1, 6) + pad(crop_label(row["crop"]), 34)
              + pad(f"{row['index']:.2f}", 8, "right")
              + pad(f"{row['pred']:.2f} {unit}", 18, "right")
              + pad(f"{row['median_yield']:.2f} {unit}", 18, "right"))
    worst = ranking.iloc[-1]
    print(f"\n  피하는 것이 좋은 작물: {crop_label(worst['crop'])} (적합도 {worst['index']:.2f})")
    print("  적합도 = 예상 수확량 ÷ 그 작물의 보통 수확량 (1.0 = 평년 수준)")


MENU["2"] = ("토양 맞춤 작물 추천 (TOP 5)", feature_recommend_crops)


# ===========================================================================
# 기능 3. 토양 개선 시뮬레이션
#   고른 작물에 대해 토양 성분을 하나씩 조금씩 바꿔 가며 다시 예측하고,
#   수확량이 가장 많이 오르는 조정 방향을 알려 준다. (기상은 사람이 바꿀 수 없으므로 고정)
# ===========================================================================
SOIL_STEPS = {            # 한 번에 시험해 볼 조정 폭
    "n": [-20, -10, 10, 20],
    "p": [-10, -5, 5, 10],
    "k": [-10, -5, 5, 10],
    "ph": [-0.5, -0.3, 0.3, 0.5],
}
SOIL_ACTIONS = {          # 조정 방향별 일반적인 관리 방법 (참고용 안내)
    ("n", +1): "질소질 비료(요소 등)나 녹비작물로 질소 보충",
    ("n", -1): "질소 비료 사용량 줄이기",
    ("p", +1): "인산질 비료(용과린 등) 시비",
    ("p", -1): "인산 비료 사용량 줄이기",
    ("k", +1): "칼리질 비료(염화칼륨 등) 시비",
    ("k", -1): "칼리 비료 사용량 줄이기",
    ("ph", +1): "석회(고토석회)를 뿌려 산성 토양 교정",
    ("ph", -1): "황 가루·유기물로 알칼리 토양 교정",
}


def feature_soil_simulation(model, reference, stats, store):
    print("\n━━ 3. 토양 개선 시뮬레이션 ━━")
    print("개선 효과를 보고 싶은 작물을 골라 주세요.")
    crop = choose_crop(stats)
    print("재배철을 골라 주세요.")
    season = choose_season(stats, crop)
    conditions = store.get()
    ranges = reference["input_ranges"]

    # 현재 조건 + 조정 시나리오들을 한꺼번에 예측
    scenarios = [("현재 토양", None, 0, conditions)]
    for col, steps in SOIL_STEPS.items():
        lo, hi = ranges[col]
        for step in steps:
            new_value = round(conditions[col] + step, 2)
            if lo <= new_value <= hi:           # 학습 데이터 범위 밖으로는 시험하지 않음
                changed = {**conditions, col: new_value}
                scenarios.append((f"{FEATURE_LABELS[col]} {step:+g}", col, step, changed))
    rows = pd.DataFrame([{**s[3], "crop": crop, "season": season} for s in scenarios])
    preds = model.predict(rows)
    base = preds[0]
    unit = yield_unit(crop)

    results = pd.DataFrame({"시나리오": [s[0] for s in scenarios], "col": [s[1] for s in scenarios],
                            "step": [s[2] for s in scenarios], "pred": preds})
    results["변화율"] = results["pred"] / base - 1
    better = results.iloc[1:].sort_values("변화율", ascending=False)

    print(f"\n▶ {crop_label(crop)} · 현재 예상 수확량 {base:.2f} {unit}")
    print("  " + pad("조정", 24) + pad("예상 수확량", 16, "right") + pad("변화", 10, "right"))
    for _, r in better.head(5).iterrows():
        print("  " + pad(r["시나리오"], 24) + pad(f"{r['pred']:.2f} {unit}", 16, "right")
              + pad(f"{r['변화율']:+.1%}", 10, "right"))

    best = better.iloc[0]
    if best["변화율"] < 0.01:
        print("\n  현재 토양 조건이 이 작물에 이미 알맞은 편입니다. (1% 이상 오르는 조정 없음)")
    else:
        direction = 1 if best["step"] > 0 else -1
        print(f"\n  가장 효과적인 조정: {best['시나리오']} → 수확량 {best['변화율']:+.1%}")
        print(f"  관리 방법(일반적인 예): {SOIL_ACTIONS[(best['col'], direction)]}")
    print("  ※ 학습 데이터의 토양값은 주(州) 단위 평균(30가지)이라, 이 결과는 인과관계가 아닌")
    print("    '경향'입니다. 실제 비료량은 토양 검정 기관의 시비 처방서를 따라 주세요.")


MENU["3"] = ("토양 개선 시뮬레이션", feature_soil_simulation)


# ===========================================================================
# 메인
# ===========================================================================
def main():
    model, reference, stats = load_resources()
    store = ConditionStore(reference["input_ranges"])
    m = reference["test_metrics"]
    print("=" * 60)
    print(" 토양 조건별 작물 추천 프로그램")
    print(f" 모델: {reference['algorithm']} · {reference['test_period']}년 평가 R² {m['R2']:.3f}")
    print(" 학습 데이터: 인도 30개 주 작물 통계(1997~2020). 다른 지역·단위의 토양 검정값에는")
    print(" 오차가 클 수 있으니 참고용으로 활용해 주세요.")
    print("=" * 60)
    while True:
        print("\n[메뉴]")
        for key, (name, _) in MENU.items():
            print(f"  {key}. {name}")
        print("  0. 종료")
        key = input("번호를 선택하세요: ").strip()
        if key == "0":
            print("프로그램을 종료합니다.")
            break
        if key not in MENU:
            print("메뉴에 있는 번호를 입력해 주세요.")
            continue
        MENU[key][1](model, reference, stats, store)


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 취소되었습니다.")
