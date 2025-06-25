# 🚢 타이타닉 생존자 예측 모델 (XGBoost 활용)

---

## 1. 프로젝트 개요

이 프로젝트는 1912년 침몰한 타이타닉호 승객들의 다양한 정보를 바탕으로 **생존 여부를 예측하는 머신러닝 모델을 개발**하는 것을 목표로 합니다. Kaggle의 'Titanic - Machine Learning from Disaster' 경진대회 데이터를 활용하여 데이터 분석, 전처리, 모델링, 평가의 전 과정을 수행했습니다. 특히, 강력한 성능을 자랑하는 **XGBoost 분류기**를 사용하여 예측 정확도를 높였습니다.

---

## 2. 사용 기술

* **언어:** Python
* **라이브러리:**
    * `pandas`: 데이터 처리 및 분석
    * `numpy`: 수치 계산
    * `matplotlib`, `seaborn`: 데이터 시각화 (EDA)
    * `scikit-learn`: 데이터 전처리, 모델 평가, 교차 검증, 하이퍼파라미터 튜닝
    * `xgboost`: 핵심 예측 모델 (XGBoost Classifier)
* **개발 환경:** Google Colab (CPU 런타임 최적화)

---

## 3. 데이터셋

* **출처:** Kaggle - [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)
* **파일:** `train.csv`, `test.csv`
* **주요 컬럼:**
    * `Survived`: 생존 여부 (0 = 사망, 1 = 생존) - **타겟 변수**
    * `Pclass`: 객실 등급 (1, 2, 3)
    * `Sex`: 성별
    * `Age`: 나이
    * `SibSp`: 함께 탑승한 형제/배우자 수
    * `Parch`: 함께 탑승한 부모/자녀 수
    * `Fare`: 요금
    * `Embarked`: 승선 항구 (C, Q, S)

---

## 4. 프로젝트 진행 과정

### 4.1. 데이터 불러오기 및 탐색 (EDA)

* `train.csv`와 `test.csv` 파일 로드
* 데이터의 기본 정보 (`.info()`, `.describe()`, `.head()`) 확인
* 컬럼별 결측치 확인 (`.isnull().sum()`)
* `Survived` 컬럼 분포 확인 및 주요 컬럼(예: `Sex`, `Pclass`, `Age`, `Fare`)과 생존율 간의 관계 시각화 (`countplot`, `barplot`, `histplot` 등)

### 4.2. 데이터 전처리 및 특징 공학

* **불필요 컬럼 제거:** `PassengerId`, `Name`, `Ticket`, `Cabin` 컬럼 제거 (단, `test_df`의 `PassengerId`는 제출용으로 별도 저장)
* **결측치 처리:**
    * `Age` 컬럼: `Pclass`와 `Sex` 조합별 **중앙값**으로 대체
    * `Embarked` 컬럼: **최빈값**으로 대체
    * `Fare` 컬럼 (test_df): **중앙값**으로 대체
* **범주형 데이터 인코딩:**
    * `Sex` 컬럼: 이진 인코딩 (`male`: 0, `female`: 1)
    * `Embarked` 컬럼: **원-핫 인코딩** (`pd.get_dummies`)
* **특징 공학 (Feature Engineering):**
    * `FamilySize` 컬럼 생성: `SibSp` + `Parch` + 1 (본인 포함 가족 규모)
    * `IsAlone` 컬럼 생성: `FamilySize`가 1인 경우 1, 그렇지 않은 경우 0 (단독 탑승 여부)
    * `FarePerPerson` 컬럼 생성: `Fare` / `FamilySize` (1인당 요금)

### 4.3. 모델 선택 및 학습

* **모델:** XGBoost Classifier
* **학습 데이터:** 전처리된 `train.csv` 데이터 (`X_train`, `y_train`)
* **교차 검증:** 5-Fold Cross-validation을 통해 모델의 일반화 성능 검증 (평균 정확도 확인)
* **성능 지표:** 정확도 (Accuracy), 분류 보고서 (Precision, Recall, F1-score), 혼동 행렬 (Confusion Matrix)

### 4.4. 모델 최적화 및 최종 예측

* **하이퍼파라미터 튜닝:** `GridSearchCV`를 사용하여 `n_estimators`, `learning_rate`, `max_depth`에 대한 최적의 조합 탐색
    * **최적 파라미터:** (예시: `{'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 200}`)
    * **최고 교차 검증 정확도:** (예시: `0.82xx`)
* **최종 예측:** 최적화된 XGBoost 모델로 전처리된 `test.csv` 데이터에 대한 생존 여부 예측 수행
* **제출 파일 생성:** Kaggle 제출 형식(`PassengerId`, `Survived`)에 맞춰 `submission.csv` 파일 생성

---

## 5. 결과 및 결론

* 최종 모델은 **약 75.837%의 정확도** (Kaggle Public Leaderboard 점수 기입)로 타이타닉 승객의 생존 여부를 예측했습니다.
* 데이터 탐색 결과, **성별(여성 생존율 높음)**과 **객실 등급(1등급 승객 생존율 높음)**이 생존에 가장 큰 영향을 미치는 요인임을 확인할 수 있었습니다. 이는 모델의 예측 결과에서도 중요하게 반영되었습니다.
* 가족 규모(`FamilySize`)와 혼자 탑승한 경우(`IsAlone`)와 같은 **새로운 특징을 생성**하여 모델의 예측 성능 향상에 기여했습니다.
* **XGBoost**는 정형 데이터 예측 문제에서 강력한 성능을 보여주었으며, **교차 검증과 하이퍼파라미터 튜닝**을 통해 모델의 안정성과 정확도를 확보할 수 있었습니다.

---

## 6. 향후 개선 방향

* **더 다양한 특징 공학:** `Name` 컬럼에서 호칭(Title)을 추출하여 새로운 특징으로 활용하거나, `Cabin` 컬럼의 정보를 더 효과적으로 활용하는 방법 모색.
* **다른 모델 시도 및 앙상블:** LightGBM, CatBoost 등 다른 부스팅 모델이나 로지스틱 회귀, SVM 등 다양한 모델을 시도해보고, 여러 모델의 예측을 결합하는 앙상블(Ensemble) 기법 적용.
* **스태킹(Stacking):** 여러 모델의 예측을 다시 학습 데이터로 활용하는 스태킹 기법을 통해 예측 성능 극대화.
* **데이터 불균형 해결:** 생존자와 사망자 수의 불균형을 해결하기 위한 오버샘플링(Over-sampling) 또는 언더샘플링(Under-sampling) 기법 적용.

---
