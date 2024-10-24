import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
sns.set(style="whitegrid")
# plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['font.family'] = 'NanumBarunGothic'
# 음수 기호 설정
plt.rcParams['axes.unicode_minus'] = False  # 음수 기호를 제대로 표시하게 설정
import shap
shap.initjs()
from catboost import CatBoostRegressor  # 추가된 import
import pickle


def save_model_artifacts(df, df_train_X, df_train_y, df_test_X, df_test_y, catboost_model, 
                        price_tr_scaler, scaler, label_encoders, model_metrics):
    """모델 학습 결과물 저장"""
    # 저장할 디렉토리 생성
    if not os.path.exists('model_artifacts'):
        os.makedirs('model_artifacts')
    
    # 1. 전처리된 데이터 저장
    processed_data = {
        'df': df,
        'train_X': df_train_X,
        'train_y': df_train_y,
        'test_X': df_test_X,
        'test_y': df_test_y
    }
    with open('model_artifacts/processed_data.pkl', 'wb') as f:
        pickle.dump(processed_data, f)
    
    # 2. 스케일러와 인코더 저장
    preprocessing_objects = {
        'price_scaler': price_tr_scaler,
        'feature_scaler': scaler,
        'label_encoders': label_encoders
    }
    with open('model_artifacts/preprocessing_objects.pkl', 'wb') as f:
        pickle.dump(preprocessing_objects, f)
    
    # 3. 모델 성능 지표 저장
    with open('model_artifacts/model_metrics.pkl', 'wb') as f:
        pickle.dump(model_metrics, f)
    
    # 4. CatBoost 모델 저장
    catboost_model.save_model('model_artifacts/catboost_model.cbm')
    
    # 5. SHAP 값 계산 및 저장
    sample_size = min(1000, len(df_train_X))
    sample_indices = np.random.choice(len(df_train_X), sample_size, replace=False)
    sample_data = df_train_X.iloc[sample_indices]
    
    explainer = shap.TreeExplainer(catboost_model)
    shap_values = explainer.shap_values(sample_data)
    
    shap_data = {
        'values': shap_values,
        'feature_names': df_train_X.columns,
        'sample_data': sample_data
    }
    with open('model_artifacts/shap_values.pkl', 'wb') as f:
        pickle.dump(shap_data, f)

# 메인 코드
if __name__ == "__main__":
    # 데이터 로드
    df = pd.read_excel('raw_data_241023.xlsx', engine='openpyxl')
    
    # 불필요한 컬럼 제거
    df.drop(['category','latitude', 'longitude'],axis=1, inplace=True)
    df.drop(['지하1층','1층','2층','3층','4층', '5층','6-10층','11층이상'],axis=1, inplace=True)
    df.drop(['gfa_crr','class_dlt ','class_type'],axis=1, inplace=True)

    # 결측치 처리
    grouped_avg = df.groupby(['dealt_yr', 'dealt_qr'])['vacancy_rate'].transform('mean')
    df['vacancy_rate']=df['vacancy_rate'].fillna(grouped_avg)
    df = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != 'object' else x)

    # 새로운 특성 생성
    df['cap_rate'] = (df['순영업소득(천원/㎡)'] * df['gfa_dlt']) / df['price_tr'] * 100
    df['total_operating_income'] = df['순영업소득(천원/㎡)'] * df['gfa_dlt']

    # 데이터 분할
    df_train = df[df.dealt_yr!=2024]
    df_test = df[df.dealt_yr==2024]

    df_train_X = df_train.drop(columns=['price_tr','투자수익률','기타수입(%)','순영업소득(천원/㎡)','운영경비(%)','add_dong','add_si'])
    df_train_y = df_train['price_tr']
    df_test_X = df_test.drop(columns=['price_tr','투자수익률','기타수입(%)','순영업소득(천원/㎡)','운영경비(%)','add_dong','add_si'])
    df_test_y = df_test['price_tr']

    # 레이블 인코딩
    label_encoders = {}
    
    # dealt_qr 처리
    df_train_X['dealt_qr'] = df_train_X['dealt_qr'].str.replace('Q', '').astype(int)
    df_test_X['dealt_qr'] = df_test_X['dealt_qr'].str.replace('Q', '').astype(int)

    # 범주형 변수 처리
    categorical_columns = df_train_X.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_train_X[col] = le.fit_transform(df_train_X[col].astype(str))
        label_encoders[col] = le
        df_test_X[col] = label_encoders[col].transform(df_test_X[col].astype(str))

    # 수치형 변수 스케일링
    numeric_columns = df_train_X.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df_train_X[numeric_columns] = scaler.fit_transform(df_train_X[numeric_columns])
    df_test_X[numeric_columns] = scaler.transform(df_test_X[numeric_columns])

    # 종속변수 스케일링
    price_tr_scaler = StandardScaler()
    df_train_y_scaled = price_tr_scaler.fit_transform(df_train_y.values.reshape(-1, 1))
    df_test_y_scaled = price_tr_scaler.transform(df_test_y.values.reshape(-1, 1))

    # Train/Validation Split
    X_train, X_val, y_train, y_val = train_test_split(
        df_train_X, 
        df_train_y_scaled, 
        test_size=0.3, 
        random_state=42
    )

    # CatBoost 모델 학습
    catboost_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.2,
        depth=6,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=24,
        verbose=100,
        early_stopping_rounds=10
    )

    # 범주형 변수 인덱스 제거 (이미 인코딩되었으므로)
    catboost_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )

    # 성능 평가
    y_pred_val = catboost_model.predict(X_val)
    y_pred_val = price_tr_scaler.inverse_transform(y_pred_val.reshape(-1, 1))
    y_val_actual = price_tr_scaler.inverse_transform(y_val)

    # 테스트 세트 예측
    y_pred_test = catboost_model.predict(df_test_X)
    y_pred_test = price_tr_scaler.inverse_transform(y_pred_test.reshape(-1, 1))

    # 모델 성능 지표 계산
    model_metrics = {
        'validation': {
            'mape': mean_absolute_percentage_error(y_val_actual, y_pred_val),
            'rmse': np.sqrt(mean_squared_error(y_val_actual, y_pred_val)),
            'mae': mean_absolute_error(y_val_actual, y_pred_val)
        },
        'test': {
            'mape': mean_absolute_percentage_error(df_test_y, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(df_test_y, y_pred_test)),
            'mae': mean_absolute_error(df_test_y, y_pred_test)
        }
    }

    # 모든 결과물 저장
    save_model_artifacts(
        df, df_train_X, df_train_y, df_test_X, df_test_y,
        catboost_model, price_tr_scaler, scaler, label_encoders,
        model_metrics
    )

    # 성능 출력
    print("\nValidation Metrics:")
    print(f"MAPE: {model_metrics['validation']['mape']:.4f}")
    print(f"RMSE: {model_metrics['validation']['rmse']:.4f}")
    print(f"MAE: {model_metrics['validation']['mae']:.4f}")

    print("\nTest Metrics:")
    print(f"MAPE: {model_metrics['test']['mape']:.4f}")
    print(f"RMSE: {model_metrics['test']['rmse']:.4f}")
    print(f"MAE: {model_metrics['test']['mae']:.4f}")