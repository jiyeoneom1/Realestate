# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import shap
from catboost import CatBoostRegressor
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import matplotlib.font_manager as fm  # 폰트 관련 모듈

# 페이지 설정
st.set_page_config(
    page_title="부동산 투자 분석 플랫폼",
    page_icon="🏢",
    layout="wide"
)

def load_font():
    font_path = os.path.join(os.getcwd(), 'fonts', 'NanumBarunGothic.ttf')  # 폰트 경로 설정
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()  # matplotlib에 폰트 설정
    
# 저장된 데이터 및 모델 로드
@st.cache_resource
def load_artifacts():
    with open('model_artifacts/processed_data.pkl', 'rb') as f:
        processed_data = pickle.load(f)
    
    with open('model_artifacts/preprocessing_objects.pkl', 'rb') as f:
        preprocessing_objects = pickle.load(f)
    
    with open('model_artifacts/model_metrics.pkl', 'rb') as f:
        model_metrics = pickle.load(f)
    
    with open('model_artifacts/shap_values.pkl', 'rb') as f:
        shap_data = pickle.load(f)
    
    model = CatBoostRegressor()
    model.load_model('model_artifacts/catboost_model.cbm')
    
    return processed_data, preprocessing_objects, model_metrics, shap_data, model

def main():
    # 데이터 로드
    processed_data, preprocessing_objects, model_metrics, shap_data, model = load_artifacts()
    df = processed_data['df']
    
    # 사이드바
    with st.sidebar:
        menu = option_menu(
            '메뉴',
            ['데이터 분석', '모델 성능', '투자 분석', 'SHAP 분석'],  # 메뉴 항목
            icons=['bar-chart-line-fill', 'bar-chart', 'graph-up-arrow', 'gear'],  # 각 메뉴의 아이콘
            menu_icon="caret-down-fill",  # 메뉴 아이콘
            default_index=0,
            styles={
                "container": {"padding": "5!important", "background-color": "#fafafa"},  # 메뉴 컨테이너 스타일
                "icon": {"color": "#243746", "font-size": "25px"},  # 아이콘 스타일
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},  # 메뉴 링크 스타일
                "nav-link-selected": {"background-color": "#ef494c"},  # 선택된 항목의 스타일
            }
        )
    
    if menu == "데이터 분석":
        st.title("업무용 부동산 데이터 분석")
        
        # 필터 설정
        col1, col2 = st.columns(2)
        with col1:
            years = sorted(df['dealt_yr'].unique())
            selected_years = st.multiselect(
                "연도 선택",
                options=years,
                default=years
            )
        
        # 필터링된 데이터
        filtered_df = df[df['dealt_yr'].isin(selected_years)]
        
        # 주요 지표
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "평균 거래가격",
                f"{filtered_df['price_tr'].mean():,.0f}백만원",
                f"{filtered_df['price_tr'].mean() - df['price_tr'].mean():,.0f}"
            )
        with col2:
            st.metric(
                "평균 임대료",
                f"{filtered_df['rent'].mean():,.0f}원/㎡",
                f"{filtered_df['rent'].mean() - df['rent'].mean():,.0f}"
            )
        with col3:
            st.metric(
                "평균 공실률",
                f"{filtered_df['vacancy_rate'].mean():.1f}%", 
                f"{(filtered_df['vacancy_rate'].mean() - df['vacancy_rate'].mean()):.1f}%p"
            )
        
        # 거래가격 트렌드
        st.subheader("거래가격 트렌드")
        yearly_data = filtered_df.groupby('dealt_yr')['price_tr'].agg(['mean', 'std']).reset_index()
        yearly_data.columns = ['year', 'mean', 'std']

        fig = go.Figure()

        # 평균 거래가격 라인
        fig.add_trace(go.Scatter(
            x=yearly_data['year'],
            y=yearly_data['mean'],
            mode='lines+markers',
            name='평균 거래가격',
            line=dict(color='royalblue')
        ))

        # 오차 범위
        fig.add_trace(go.Scatter(
            x=yearly_data['year'],
            y=yearly_data['mean'] + yearly_data['std'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=yearly_data['year'],
            y=yearly_data['mean'] - yearly_data['std'],
            mode='lines',
            name='Lower Bound',
            line=dict(width=0),
            fillcolor='rgba(68, 68, 68, 0.2)',
            fill='tonexty',
            showlegend=False
        ))

        fig.update_layout(
            title='연도별 평균 거래가격 추이',
            xaxis_title='연도',
            yaxis_title='거래가격 (백만원)',
            hovermode='x'
        )

        st.plotly_chart(fig, use_container_width=True)
        
        # 지역별 시계열 분석
        st.subheader("지역별 시계열 분석")
        selected_si = st.selectbox("시 선택", df['add_si'].unique())
        filtered_gu = df[df['add_si'] == selected_si]['add_gu'].unique()
        selected_gu = st.selectbox("구 선택", filtered_gu)

        region_data = df[(df['add_si'] == selected_si) & (df['add_gu'] == selected_gu)]
        region_data = region_data[['dealt_yr', 'price_tr']].groupby('dealt_yr').mean()

        fig, ax = plt.subplots(figsize=(8, 4))  # 가로 8, 세로 4로 설정
        ax.plot(region_data.index, region_data['price_tr'], marker='o')
        ax.set_title(f"Annual price trend in {selected_si} {selected_gu} area")
        ax.set_xlabel('Year')
        ax.set_ylabel('average transaction price(millions)')
        st.pyplot(fig)

        # 상관관계 분석
        st.subheader("주요 변수 상관관계")
        
        # 변수명 매핑
        var_names = {
            'price_tr': '거래가격',
            'rent': '임대료',
            'vacancy_rate': '공실률',
            'total_operating_income': '총영업수익',
            'cap_rate': '자본수익률'
        }

        numeric_cols = ['price_tr', 'rent', 'vacancy_rate', 'total_operating_income', 'cap_rate']
        corr = filtered_df[numeric_cols].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr,
            x=[var_names[col] for col in numeric_cols],
            y=[var_names[col] for col in numeric_cols],
            text=corr.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            colorscale='RdBu',
            zmid=0
        ))

        fig.update_layout(
            title='변수간 상관관계',
            xaxis_title='변수',
            yaxis_title='변수',
            width=700,
            height=700
        )

        st.plotly_chart(fig, use_container_width=True)
        
    elif menu == "모델 성능":
        st.title("모델 성능 평가")
        
        # 성능 지표 표시
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "MAPE",
                f"{model_metrics['test']['mape']:.2%}",
                f"{model_metrics['test']['mape'] - model_metrics['validation']['mape']:.2%}p"
            )
        with col2:
            st.metric(
                "RMSE",
                f"{model_metrics['test']['rmse']:,.0f}",
                f"{model_metrics['test']['rmse'] - model_metrics['validation']['rmse']:,.0f}"
            )
        with col3:
            st.metric(
                "MAE",
                f"{model_metrics['test']['mae']:,.0f}",
                f"{model_metrics['test']['mae'] - model_metrics['validation']['mae']:,.0f}"
            )
        
        # 예측값 vs 실제값 비교
        st.subheader("예측값 vs 실제값")
        y_pred = model.predict(processed_data['test_X'])
        y_pred = preprocessing_objects['price_scaler'].inverse_transform(y_pred.reshape(-1, 1))
        y_true = processed_data['test_y']
        
        fig = px.scatter(
            x=y_true,
            y=y_pred.flatten(),
            labels={'x': '실제값', 'y': '예측값'},
            title='실제값 vs 예측값 비교'
        )
        fig.add_trace(
            go.Scatter(
                x=[y_true.min(), y_true.max()],
                y=[y_true.min(), y_true.max()],
                mode='lines',
                name='y=x',
                line=dict(dash='dash')
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif menu == "투자 분석":
        st.title("부동산 투자 분석: 지도 기반 시나리오 분석")
        
        # 연도 선택
        years = sorted(df['dealt_yr'].unique())
        selected_year = st.selectbox("연도 선택", years)
        # 시 선택
        si_options = df['add_si'].unique()
        selected_si = st.selectbox("시 선택", si_options)

        # 선택한 시에 해당하는 구 옵션 필터링
        gu_options = df[df['add_si'] == selected_si]['add_gu'].unique()
        selected_gu = st.selectbox("구 선택", gu_options)

        # 시나리오 분석
        st.subheader("시나리오 분석")
        scenario_dict = {
            '낙관적': {'rent': 1.05, 'vacancy_rate': 0.95, 'cap_rate': 1.1, 'color': 'green'},
            '중립적': {'rent': 1.0, 'vacancy_rate': 1.0, 'cap_rate': 1.0, 'color': 'orange'},
            '비관적': {'rent': 0.95, 'vacancy_rate': 1.1, 'cap_rate': 0.9, 'color': 'red'}
        }

        filtered_data = df[(df['dealt_yr'] == selected_year) & (df['add_si'] == selected_si) & (df['add_gu'] == selected_gu)]
        filtered_data['lat'] = np.random.uniform(low=37.4, high=37.6, size=len(filtered_data))
        filtered_data['lon'] = np.random.uniform(low=126.8, high=127.0, size=len(filtered_data))
        
        fig = px.scatter_mapbox(
            filtered_data,
            lat='lat',
            lon='lon',
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'],
            mapbox_style="carto-positron",
            zoom=10,
            title=f"{selected_year}년 {selected_si} {selected_gu} 시나리오별 분석",
            size_max=15,
            height=600)

        for scenario, values in scenario_dict.items():
            scenario_data = filtered_data.copy()
            scenario_data['adjusted_rent'] = scenario_data['rent'] * values['rent']
            scenario_data['adjusted_vacancy_rate'] = scenario_data['vacancy_rate'] * values['vacancy_rate']
            scenario_data['adjusted_cap_rate'] = scenario_data['cap_rate'] * values['cap_rate']
            fig.add_trace(
                go.Scattermapbox(
                    lat=scenario_data['lat'],
                    lon=scenario_data['lon'],
                    mode='markers',
                    marker=go.scattermapbox.Marker(size=14, color=values['color']),
                    text=[f"{scenario} 시나리오 예상 가격: {price}" for price in scenario_data['price_tr']],
                    name=f"{scenario} 시나리오"
                )
            )
        
        st.plotly_chart(fig)
                 
    elif menu == "SHAP 분석":
        st.title("SHAP 중요도 분석")
        
        st.write("특성 중요도 시각화")
        
        # SHAP 값 시각화
        shap_values = shap_data['values']
        feature_names = shap_data['feature_names']
        sample_data = shap_data['sample_data']
        
        # 특성 중요도 그래프
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            sample_data,
            feature_names=feature_names,
            show=False
        )
        st.pyplot(fig)
        
        # 개별 특성의 SHAP 값 분포
        st.subheader("주요 특성별 영향도")
        selected_feature = st.selectbox(
            "특성 선택",
            options=feature_names
        )
        
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(
            selected_feature,
            shap_values,
            sample_data,
            ax=ax,
            show=False
        )
        st.pyplot(fig)

if __name__ == "__main__":
    main()
