"""원시 데이터 수집·저장 스크립트 (5장: 파일 형태의 오픈 데이터 수집, sqlite3 저장)

1) 인도 주(州)별 작물 생산 통계, 2) 주별 토양 성분, 3) 주별·연도별 기상 자료를 내려받아
   data/raw/ 에 CSV로 보관하고, data/crop.db (SQLite)의 테이블로 적재한다.

데이터 출처
- 작물 생산 통계(crop_yield.csv): Mendeley Data "Stacked Ensemble Model for Accurate Crop Yield
  Prediction" (doi:10.17632/ncw2vbcgnk), 인도 정부 농업통계 기반, 1997~2020년, CC BY 4.0
- 주별 토양·기상 표: GitHub vuongnd28/crop-yield-prediction 저장소 작성자가 정리한 참고값
  (원 출처는 저장소에 명시되어 있지 않음 → 보고서에 한계로 기재)
"""

import sqlite3
import urllib.request
from io import BytesIO

import pandas as pd

from common import DB_PATH, RAW_DIR

BASE_URL = "https://raw.githubusercontent.com/vuongnd28/crop-yield-prediction/main/data/"
FILES = {
    # 테이블 이름: (파일 이름, 꼭 있어야 하는 열)
    "crop_yield": ("crop_yield.csv",
                   ["crop", "year", "season", "state", "area", "production", "yield"]),
    "state_soil": ("state_soil_data.csv", ["state", "N", "P", "K", "pH"]),
    "state_weather": ("state_weather_data_1997_2020.csv",
                      ["state", "year", "avg_temp_c", "total_rainfall_mm",
                       "avg_humidity_percent"]),
}


def download_bytes(file_name):
    """원본 파일을 바이트 그대로 받는다 (원시 데이터는 손대지 않고 보관하기 위해)"""
    request = urllib.request.Request(BASE_URL + file_name,
                                     headers={"User-Agent": "soil-crop-recommender (student project)"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)  # DB 파일 열기 (없으면 새로 만듦)
    try:
        for table, (file_name, required) in FILES.items():
            local = RAW_DIR / file_name
            if local.is_file():
                print(f"[{table}] 이미 받은 파일 사용: {local.name}")
                df = pd.read_csv(local)
            else:
                print(f"[{table}] 내려받는 중: {BASE_URL + file_name}")
                try:
                    content = download_bytes(file_name)
                except Exception as error:
                    raise SystemExit(f"내려받기 실패: {error}\n인터넷 연결을 확인해 주세요.")
                local.write_bytes(content)
                df = pd.read_csv(BytesIO(content))

            missing = [c for c in required if c not in df.columns]
            if missing:
                raise SystemExit(f"{file_name} 형식이 예상과 다릅니다. 없는 열: {missing}")

            df.to_sql(table, conn, if_exists="replace", index=False)
            print(f"   → {table} 테이블 저장: {len(df):,}행 × {df.shape[1]}열, "
                  f"결측치 {int(df.isnull().sum().sum())}개")

        # 저장이 잘 되었는지 SQL로 다시 읽어 확인
        check = pd.read_sql(
            "SELECT COUNT(*) AS 행수, COUNT(DISTINCT TRIM(crop)) AS 작물수, "
            "MIN(year) AS 시작연도, MAX(year) AS 마지막연도 FROM crop_yield", conn)
        print("[확인]", check.to_dict(orient="records")[0])
        print(f"저장 완료: {DB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
