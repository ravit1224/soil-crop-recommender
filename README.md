# 토양 조건별 작물 추천 (Soil-based Crop Recommender)

토양 성분(N·P·K·pH)과 기상 조건을 입력하면, 머신러닝 회귀 모델이 작물별 예상 수확량을 예측해
그 땅에 알맞은 작물을 추천하는 터미널(CLI) 프로그램입니다.

> 고려대학교 「데이터과학과인공지능」 과제: 데이터 기반 예측 모델 및 서비스 개발하기

## 실행 순서

```
pip install -r requirements.txt
python collect_data.py   # 1) 원시 데이터 내려받기 → data/raw/, data/crop.db
python train_model.py    # 2) 전처리·모델 비교·튜닝·평가 → models/, figures/
```

`train_model.py`를 실행하면 학습 과정이 단계별로 출력됩니다. 제가 실행했을 때의 출력은
[docs/train_log.txt](docs/train_log.txt)에 그대로 저장해 두었습니다.
