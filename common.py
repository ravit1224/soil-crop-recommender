"""모델 생성 코드(train_model.py)와 응용 SW(app.py)가 함께 쓰는 설정값"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
DB_PATH = BASE_DIR / "data" / "crop.db"            # 5장: sqlite3로 원시 데이터 저장·관리
MODEL_PATH = BASE_DIR / "models" / "crop_yield_model.joblib"
REFERENCE_PATH = BASE_DIR / "models" / "crop_reference.json"
FIGURE_DIR = BASE_DIR / "figures"

RANDOM_STATE = 42
TEST_START_YEAR = 2017       # 2017~2020년을 평가 데이터로 사용 (미래 연도 예측)
MIN_ROWS_PER_CROP = 100      # 관측치가 이보다 적은 작물은 학습·추천에서 제외

# ---------------------------------------------------------------------------
# 모델 입력 피쳐: 서로 정보가 겹치지 않도록 고른 9개
#   - 토양 조건 4개  : 사용자가 토양 검정으로 알 수 있는 값
#   - 기상 조건 3개  : 재배 기간의 기후
#   - 재배 정보 2개  : 어떤 작물을, 어떤 재배철에
# 제외한 열과 이유는 train_model.py 의 preprocess() 주석과 보고서에 정리
# ---------------------------------------------------------------------------
SOIL_FEATURES = ["n", "p", "k", "ph"]
WEATHER_FEATURES = ["avg_temp_c", "total_rainfall_mm", "avg_humidity_percent"]
NUMERIC_FEATURES = SOIL_FEATURES + WEATHER_FEATURES
CATEGORICAL_FEATURES = ["crop", "season"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "yield"

FEATURE_LABELS = {
    "n": "토양 질소(N)",
    "p": "토양 인(P)",
    "k": "토양 칼륨(K)",
    "ph": "토양 산도(pH)",
    "avg_temp_c": "연평균 기온(℃)",
    "total_rainfall_mm": "연 강수량(mm)",
    "avg_humidity_percent": "연평균 습도(%)",
}

# 집계 항목(Other ~, ~ total)은 실제 작물이 아니므로 추천 대상에서 제외한다.
# 코코넛은 수확량 단위가 '개수/ha'로 다른 작물(톤/ha)과 달라 제외한다.
EXCLUDED_CROPS = [
    "Other Kharif pulses", "Other  Rabi pulses", "Other Rabi pulses", "Other Cereals",
    "other oilseeds", "Oilseeds total", "Other Summer Pulses", "Coconut",
]

CROP_NAMES_KO = {
    "Rice": "벼", "Maize": "옥수수", "Wheat": "밀", "Barley": "보리",
    "Jowar": "수수(조와르)", "Bajra": "진주조(바즈라)", "Ragi": "손가락조(라기)",
    "Small millets": "소립 잡곡(기장류)", "Moong(Green Gram)": "녹두",
    "Urad": "흑녹두(우라드)", "Arhar/Tur": "비둘기콩", "Gram": "병아리콩",
    "Masoor": "렌틸콩", "Moth": "모스콩", "Horse-gram": "말콩(호스그램)",
    "Cowpea(Lobia)": "동부콩", "Peas & beans (Pulses)": "완두·강낭콩",
    "Soyabean": "대두(콩)", "Groundnut": "땅콩", "Sesamum": "참깨",
    "Rapeseed &Mustard": "유채·겨자", "Sunflower": "해바라기", "Linseed": "아마씨",
    "Castor seed": "피마자", "Safflower": "잇꽃(홍화)", "Niger seed": "니게르씨",
    "Potato": "감자", "Sweet potato": "고구마", "Tapioca": "카사바",
    "Onion": "양파", "Garlic": "마늘", "Ginger": "생강", "Turmeric": "강황",
    "Dry chillies": "건고추", "Coriander": "고수", "Black pepper": "후추",
    "Banana": "바나나", "Cashewnut": "캐슈넛", "Arecanut": "빈랑",
    "Sugarcane": "사탕수수", "Tobacco": "담배", "Cotton(lint)": "목화",
    "Jute": "황마", "Mesta": "메스타(케나프)", "Sannhamp": "선헴프",
    "Khesari": "케사리콩", "Cardamom": "카다멈", "Guar seed": "구아르콩",
}

SEASON_NAMES_KO = {
    "Kharif": "카리프(여름 우기작, 6~10월)",
    "Rabi": "라비(겨울 건기작, 10~3월)",
    "Whole Year": "연중 재배",
    "Summer": "여름작",
    "Autumn": "가을작",
    "Winter": "겨울작",
}

# 섬유작물은 생산량을 '베일(bale)' 단위로 집계한 원자료라 수확량 단위가 다르다.
BALE_CROPS = {"Cotton(lint)", "Jute", "Mesta", "Sannhamp"}


def crop_label(crop):
    return f"{CROP_NAMES_KO.get(crop, crop)} ({crop})"


def yield_unit(crop):
    return "베일/ha" if crop in BALE_CROPS else "톤/ha"
