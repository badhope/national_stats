"""
Web仪表盘模式
基于Streamlit的交互式数据分析界面
运行方式: streamlit run web_app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from core.scraper import NationalBureauScraper
from core.predictor import ComprehensivePredictor
from config import METRICS

# 页面配置
st.set_page_config(
    page_title="国家统计局数据分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_scraper():
    return NationalBureauScraper()


@st.cache_data(ttl=3600)
def load_data(start_year, end_year):
    scraper = get_scraper()
    return scraper.fetch_all_parallel(start_year, end_year)


def main():
    # 标题
    st.markdown('<h1 class="main-header">📊 国家统计局数据分析系统</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 侧边栏设置
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/statistics.png", width=80)
        st.title("设置面板")
        
        st.subheader("1. 时间范围")
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.number_input("起始年份", 1990, 2024, 2010)
        with col2:
            end_year = st.number_input("结束年份", 1990, 2024, 2024)
        
        st.subheader("2. 功能选择")
        analysis_mode = st.radio(
            "分析模式",
            ["📈 数据概览", "🔮 趋势预测", "📊 对比分析", "📋 数据表格"]
        )
        
        st.markdown("---")
        st.info("💡 提示：调整左侧参数后，数据将自动更新")
    
    # 加载数据
    with st.spinner("正在加载数据..."):
        data = load_data(start_year, end_year)
    
    # 主内容区
    if "概览" in analysis_mode:
        show_overview(data)
    elif "预测" in analysis_mode:
        show_prediction(data)
    elif "对比" in analysis_mode:
        show_comparison(data)
    elif "表格" in analysis_mode:
        show_tables(data)


def show_overview(data):
    st.header("数据概览仪表盘")
    
    # 核心指标卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'gdp' in data:
            latest_gdp = data['gdp']['gdp_total'].iloc[-1] / 10000
            st.metric("GDP总量", f"{latest_gdp:.1f}万亿", 
                     delta=f"{data['gdp']['gdp_growth'].iloc[-1]:.1f}%")
    
    with col2:
        if 'population' in data:
            latest_pop = data['population']['total_population'].iloc[-1] / 10000
            st.metric("总人口", f"{latest_pop:.2f}亿")
    
    with col3:
        if 'cpi' in data:
            latest_cpi = data['cpi']['cpi_yoy'].iloc[-1]
            st.metric("CPI涨幅", f"{latest_cpi}%", 
                     delta_color="inverse" if latest_cpi > 2 else "normal")
    
    with col4:
        if 'trade' in data:
            balance = data['trade']['trade_balance'].iloc[-1]
            st.metric("贸易顺差", f"{balance}亿美元")
    
    st.markdown("---")
    
    # 图表区域
    tab1, tab2 = st.tabs(["GDP分析", "人口分析"])
    
    with tab1:
        if 'gdp' in data:
            df = data['gdp']
            
            # 双轴图
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # GDP总量柱状图
            fig.add_trace(
                go.Bar(x=df['year'], y=df['gdp_total']/10000, 
                       name="GDP总量(万亿)", marker_color='#2E86AB'),
                secondary_y=False
            )
            
            # GDP增长率折线图
            fig.add_trace(
                go.Scatter(x=df['year'], y=df['gdp_growth'], 
                          mode='lines+markers', name="增长率(%)",
                          line=dict(color='#A23B72', width=3)),
                secondary_y=True
            )
            
            fig.update_layout(
                title="GDP总量与增长率变化趋势",
                hovermode='x unified',
                height=500
            )
            fig.update_yaxes(title_text="GDP总量(万亿元)", secondary_y=False)
            fig.update_yaxes(title_text="增长率(%)", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if 'population' in data:
            df = data['population']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['year'], y=df['urbanization_rate'],
                mode='lines+markers', fill='tozeroy',
                name='城镇化率', line=dict(color='#28A745')
            ))
            
            fig.update_layout(
                title="城镇化率变化趋势",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)


def show_prediction(data):
    st.header("趋势预测分析")
    
    predictor = ComprehensivePredictor()
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        metric = st.selectbox("选择预测指标", list(METRICS.keys()))
        future_years = st.slider("预测未来年数", 1, 10, 5)
        model_type = st.selectbox("预测模型", ["自动选择", "线性回归", "多项式回归", "指数增长"])
    
    with col2:
        if metric in data:
            # 获取预测结果
            predictions = predictor.predict_all_metrics(data, future_years)
            
            if metric in predictions:
                pred = predictions[metric]
                
                # 绘制预测图
                fig = go.Figure()
                
                # 历史数据
                df = data[metric]
                y_col = [c for c in df.columns if c != 'year'][0]
                
                fig.add_trace(go.Scatter(
                    x=df['year'], y=df[y_col],
                    mode='lines+markers', name='历史数据',
                    line=dict(color='#2E86AB', width=2)
                ))
                
                # 预测数据
                if 'best_result' in pred:
                    fig.add_trace(go.Scatter(
                        x=pred['best_result']['x'], y=pred['best_result']['y'],
                        mode='lines+markers', name='预测数据',
                        line=dict(color='#A23B72', width=2, dash='dash')
                    ))
                    
                    # 显示预测详情
                    st.info(f"最佳模型: **{pred['best_model']}** | R²得分: **{pred['best_result']['metrics']['r2']:.4f}**")
                
                fig.update_layout(
                    title=f"{METRICS[metric]['name']}趋势预测",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)


def show_comparison(data):
    st.header("多指标对比分析")
    
    cols = st.multiselect("选择对比指标", list(data.keys()), default=['gdp', 'population'])
    
    if len(cols) >= 2:
        # 归一化对比
        fig = go.Figure()
        
        for col in cols:
            df = data[col]
            y_col = [c for c in df.columns if c != 'year'][0]
            
            # 归一化到0-100
            normalized = (df[y_col] - df[y_col].min()) / (df[y_col].max() - df[y_col].min()) * 100
            
            fig.add_trace(go.Scatter(
                x=df['year'], y=normalized,
                mode='lines+markers', name=METRICS[col]['name']
            ))
        
        fig.update_layout(
            title="多指标归一化趋势对比",
            yaxis_title="归一化指数 (0-100)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)


def show_tables(data):
    st.header("详细数据表格")
    
    metric = st.selectbox("选择数据类型", list(data.keys()))
    
    if metric in data:
        df = data[metric]
        
        # 添加搜索功能
        search = st.text_input("搜索...")
        if search:
            df = df[df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)]
        
        st.dataframe(
            df.style.format(precision=2).background_gradient(cmap='Blues'),
            use_container_width=True,
            height=600
        )
        
        # 导出按钮
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "📥 下载CSV", csv, f"{metric}_data.csv", "text/csv"
        )


if __name__ == "__main__":
    main()
