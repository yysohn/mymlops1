import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

warnings.filterwarnings("ignore")

# 모델 저장 폴더 설정
model_dir = "credit_model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"'{model_dir}' 폴더가 생성되었습니다.")

# ── 1. 컬럼 분류 (데이터 생성 및 전처리에 사용되므로 최상단 배치) ───────────
numeric_features     = ["amount", "hour", "transaction_count_1h", "distance_from_home_km", "age"]
categorical_features = ["merchant_category", "card_type", "country"]

# ── 2. 데이터 로드 ──────────────────────────────────────────────────────────
try:
    # 파일이 있는 경우 로드
    df = pd.read_csv("credit_card_data.csv")
    print("성공: 'credit_card_data.csv' 파일을 로드했습니다.")
except FileNotFoundError:
    # 파일이 없는 경우 테스트용 가상 데이터 생성
    print("알림: 파일이 없어 테스트용 샘플 데이터를 생성합니다.")
    np.random.seed(42)
    n_samples = 500
    data = {col: np.random.uniform(0, 100, n_samples) for col in numeric_features}
    data.update({col: np.random.choice(['type_A', 'type_B', 'type_C'], n_samples) for col in categorical_features})
    data['is_fraud'] = np.random.randint(0, 2, n_samples)
    df = pd.DataFrame(data)

X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

print(f"데이터 크기: {X.shape}  |  사기 비율: {y.mean():.1%}")

# 데이터 분할 (클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 3. 전처리 서브-파이프라인 ────────────────────────────────────────────────
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer,     numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# ── 4. RandomForest 파라미터 조합 5가지 ────────────────────────────────────
param_list = [
    {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "class_weight": "balanced"},
    {"n_estimators": 200, "max_depth": 10,   "min_samples_split": 5, "class_weight": "balanced"},
    {"n_estimators": 50,  "max_depth": 5,    "min_samples_split": 10,"class_weight": None},
    {"n_estimators": 300, "max_depth": 20,   "min_samples_split": 2, "class_weight": "balanced_subsample"},
    {"n_estimators": 150, "max_depth": 8,    "min_samples_split": 4, "class_weight": "balanced"},
]

# ── 5. 파이프라인 생성 및 학습/평가 ─────────────────────────────────────────
print("\n" + "="*85)
print(f"  {'run_name':<30} {'test_acc':>9} {'train_acc':>10}  {'model_uri'}")
print("="*85)

run_results = []

for params in param_list:
    run_name = f"n{params['n_estimators']}_d{params['max_depth']}"
    
    # 파이프라인 구성 (전처리 + 모델)
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(**params, random_state=42))
    ])
    
    # 학습
    pipe.fit(X_train, y_train)

    # 평가 및 결과 저장
    acc       = accuracy_score(y_test,  pipe.predict(X_test))
    train_acc = accuracy_score(y_train, pipe.predict(X_train))
    model_uri = os.path.join(model_dir, f"pipeline_{run_name}.pkl")
    
    joblib.dump(pipe, model_uri)
    
    run_results.append({
        "run_name": run_name,
        "accuracy": acc,
        "model_uri": model_uri
    })
    
    print(f"  {run_name:<30} {acc:>9.4f} {train_acc:>10.4f}  {model_uri}")

# ── 6. 가장 좋은 모델 자동 선택 ─────────────────────────────────────────────
best = max(run_results, key=lambda x: x["accuracy"])
print(f"\n🏆 최고 모델: {best['run_name']} | accuracy: {best['accuracy']:.4f}")

# ── 7. 최고 모델 불러오기 및 상세 평가 ──────────────────────────────────────
best_pipeline_model = joblib.load(best['model_uri'])

y_pred  = best_pipeline_model.predict(X_test)
y_proba = best_pipeline_model.predict_proba(X_test)[:, 1]

print("\n" + "="*65)
print(f"테스트셋 분류 리포트 — {best['run_name']}")
print("="*65)
print(classification_report(y_test, y_pred, target_names=["정상(0)", "사기(1)"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# ── 8. 예제 데이터 예측 ──────────────────────────────────────────────────────
example_data = pd.DataFrame({
    "amount":                [500.0,    12.5],
    "hour":                  [3,        14  ],
    "transaction_count_1h":  [8,        1   ],
    "distance_from_home_km": [300.0,    2.0 ],
    "age":                   [25.0,     45.0],
    "merchant_category":     ["online", "grocery" ],
    "card_type":             ["credit", "debit"   ],
    "country":               ["foreign","domestic"],
})

# 파이프라인을 사용하면 전처리 없이 DataFrame을 바로 전달 가능
predictions = best_pipeline_model.predict(example_data)

print("\n--- 예제 데이터 예측 결과 ---")
for i, pred in enumerate(predictions):
    result = "사기(1)" if pred == 1 else "정상(0)"
    print(f"데이터 {i+1}: {result}")