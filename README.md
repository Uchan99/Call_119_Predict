# 🚨 119 신고 건수 예측 모델링 (부산광역시)

## 📌 프로젝트 개요
본 프로젝트는 2020년부터 2023년까지의 다양한 데이터를 바탕으로, **2024년 부산광역시의 행정동별 119 신고 건수(`call_count`)** 를 일 단위로 예측하는 머신러닝 모델을 개발한 것입니다.  
단순 기상 데이터뿐 아니라 **인구, 교통사고, 시정거리, 공공 날씨 등 외부 데이터를 통합**하여 **시계열 및 지역별 특성**을 정교하게 반영했습니다.

---

## 🧾 1. 사용 데이터 및 특징

### ✅ 기초 데이터
- **부산시 119 신고 데이터** (2020~2023년 일자별 행정동 기준 신고 건수)

### ✅ 외부 공공데이터 통합
- **추가 기상 데이터** (기상청, 기상산업진흥원 등)
- **행정동별 인구 데이터** (정주인구, 유동인구 포함)
- **교통사고 데이터** (일자별 사고 건수, 사망/부상)
- **시정거리(가시거리) 데이터** (관측소 기준)
- **공휴일 정보, 요일, 주말 여부 등 시간 파생 변수**

> 📌 다양한 외부 데이터를 병합함으로써 **신고 건수에 영향을 줄 수 있는 간접 요인들**을 예측 모델에 반영했습니다.

---

## 🧼 2. 데이터 전처리 및 정제

### ✅ 기본 처리
```python
df = pd.read_csv('2020_2023_최종데이터.csv')
df.drop(columns=['tm_dt', 'address_city'], inplace=True)
df['datetime'] = pd.to_datetime(df['tm'], format='%Y%m%d')
```
- 불필요한 컬럼 제거 (tm_dt, address_city)
- 날짜 컬럼을 datetime 포맷으로 변환

### ✅ 누락된 날짜-동 조합 처리
```python
# 모든 날짜-행정동 조합 생성 후 병합
full_index = pd.MultiIndex.from_product([...])
df = full_df.merge(df, on=['tm', 'address_gu', 'sub_address'], how='left')
df['call_count'] = df['call_count'].fillna(0)
```
- 존재하지 않는 날짜-행정동 조합은 신고 건수 0으로 채움

---

## 🧠 3. 피처 엔지니어링

### ✅ 시간 기반 피처 생성
```python
df['day'] = df['datetime'].dt.day
df['weekday'] = df['datetime'].dt.weekday
df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
df['month_sin'] = np.sin(2 * np.pi * df['datetime'].dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df['datetime'].dt.month / 12)
```
- 주기적 특성(월, 요일)을 sin/cos으로 인코딩
- 주말 여부, 공휴일 전후 여부도 파생

### ✅ 시계열 기반 피처
```python
df['dong_lag_1'] = df.groupby(['address_gu', 'sub_address'])['call_count'].shift(1)
df['dong_rolling_mean_7'] = df.groupby(['address_gu', 'sub_address'])['call_count'].shift(1).rolling(window=7).mean()
```
- 신고 건수의 최근 1일, 7일 평균, 표준편차 등 생성
- 해당 동 기준의 시계열 특성을 모델에 반영

### ✅ 타겟 인코딩
```python
gu_mean_map = y_train.groupby(X_train_full['address_gu']).mean()
X_train_full['address_gu_mean_target'] = X_train_full['address_gu'].map(gu_mean_map)
```
- 행정구별 평균 신고 건수를 파생 변수로 생성
- 지역별 수요 차이를 모델이 학습 가능하도록 보완

---

## 🧪 4. 검증 데이터 분할

- 시계열 특성을 고려하여 **시간 순서 기준으로 80:20 분할**
```python
X_train, X_val, y_train, y_val = train_test_split(..., test_size=0.2, shuffle=False)
```
- 검증 데이터는 약 2023년 초~말 구간의 데이터를 포함
- 학습 및 검증셋은 시간 순서 기반으로 나누어 누수 방지

## 🤖 5. 모델 구성 및 학습 방식

✅ XGBoost 회귀 모델
```python
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    tree_method='hist',
    early_stopping_rounds=30,
)
```
- 비선형성에 강한 XGBoost 회귀 모델 활용
- 학습은 전체 데이터로, 검증은 지역별로 따로 수행

### ✅ 구 단위 모델 학습
```python
for gu in gu_train.unique():
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val_gu, y_val_gu)])
```
- 각 행정구(address_gu) 단위로 개별 모델을 따로 학습
- 특정 지역에 맞는 패턴을 개별적으로 반영할 수 있음

### ✅ Optuna 기반 하이퍼파라미터 튜닝
```python
def objective(trial, ...):
    params = {
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        ...
    }
```
- Optuna 라이브러리를 통해 자동으로 최적 파라미터 탐색
- 평가 기준은 Validation RMSE 최소화

### ✅ 예측 보정
```python
scaled_pred = pred * 0.8
clipped_pred = np.clip(scaled_pred, 1, 6)
final_pred = round(clipped_pred)
```
- 과대 예측 방지를 위해 예측값을 스케일 조정 후 클리핑

## 📈 6. 성능 평가

| 항목               | 설명                                                    |
|--------------------|---------------------------------------------------------|
| **최종 RMSE (검증셋)** | 1.05 (Optuna 튜닝 후, 전체 검증 데이터 기준)              |
| **R² Score**        | 전체 검증 데이터 기준 **0.90+ 확보**                   |
| **예측값 보정 방식**| 0.8 스케일링 후 1~6 범위로 클리핑 및 반올림             |


## 🛠 7. 개발 및 실행 환경

- **개발 환경**: Google Colab (GPU 가속 없이도 실행 가능)
- **Python 버전**: 3.10 이상
- **주요 라이브러리**:
  - xgboost==1.7.6
  - scikit-learn==1.3.2
  - optuna
  - pandas
  - numpy
  - joblib
  - tqdm

