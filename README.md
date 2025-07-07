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

- 불필요한 컬럼 제거: `tm_dt`, `address_city`
- 누락된 날짜-동 조합은 **call_count = 0**으로 채움
- 외부 데이터는 날짜/행정동 단위 기준으로 병합 및 정제
- 모든 날짜는 `datetime` 변환 후 정렬

---

## 🧠 3. 피처 엔지니어링

### ✅ 시간 기반 피처
- `day`, `weekday`, `day_of_year`, `is_weekend`
- `month_sin`, `month_cos` (계절성 보완)
- 공휴일 전후 여부: `is_before_holiday`, `is_after_holiday`

### ✅ 시계열 기반 피처
- `dong_lag_1`, `dong_lag_7`
- `dong_rolling_mean_7`, `dong_rolling_std_7`
- `days_since_last_call_dong`

### ✅ 타겟 인코딩
- `address_gu`, `sub_address` 기준으로 **과거 평균 call_count**를 매핑하여 지역별 경향성 보강

### ✅ 피처 선택
- `call_count`와 **상관계수 ≤ 0.02**인 피처는 제거  
→ 모델의 과적합 방지 및 성능 개선

---

## 🧪 4. 검증 데이터 분할

- 시계열 특성을 고려하여 **시간 순서 기준으로 80:20 분할**
```python
train_test_split(..., test_size=0.2, shuffle=False)
```
- 검증 데이터는 약 2023년 초~말 구간의 데이터를 포함
- 학습 및 검증셋은 시간 순서 기반으로 나누어 누수 방지

## 🤖 5. 모델 구성 및 학습 방식

### ✅ 모델 프레임워크

- XGBRegressor (XGBoost 회귀 모델)

### ✅ 구 단위 모델 학습 구조

```python
for gu in gu_list:
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val_gu, y_val_gu)])
```
- 부산시의 각 행정구(address_gu) 단위로 개별 모델을 따로 학습
- 학습은 전체 데이터로, 검증은 해당 구의 검증 데이터로만 수행
→ 지역별 특성 반영 효과 ↑

### ✅ Optuna 기반 자동 튜닝

```python
def objective(trial, ...):
    params = {
        'max_depth': trial.suggest_int(5, 12),
        'learning_rate': trial.suggest_float(0.01, 0.2, log=True),
        ...
    }
    return RMSE
```
- Optuna로 하이퍼파라미터 최적화 수행
- Validation RMSE를 최소화하는 파라미터 탐색
- 탐색 범위: max_depth, learning_rate, gamma, lambda, alpha 등

### ✅ 예측 보정

```python
scaled_pred = pred * 0.8
clipped_pred = np.clip(scaled_pred, 1, 6)
final_pred = round(clipped_pred)
```
- 과대 예측 방지를 위해 스케일 조정(0.8) 및 클리핑(1~6) 후 반올림

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

