#!/usr/bin/env python
"""
Web应用 - Streamlit仪表盘
提供交互式的宏观经济数据分析界面
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, IndicatorLibrary, IndicatorCategory
from core.data_manager import DataManager
from core.analyzer import StatisticalAnalyzer
from core.visualizer import Visualizer
from core.reporter import ReportGenerator
from core.models import GrowthAccountingModel, OkunLawModel, PhillipsCurveModel


# ==================== 页面配置 ====================

st.set_page_config(
    page_title="宏观经济数据分析平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 全局初始化 ====================

@st.cache_resource
def init_managers():
    """初始化管理器（缓存）"""
    return {
        'data': DataManager(),
        'analyzer': StatisticalAnalyzer(),
        'visualizer': Visualizer(),
        'reporter': ReportGenerator()
    }

managers = init_managers()

# ==================== 侧边栏 ====================

st.sidebar.title("📊 宏观经济数据分析平台")
st.sidebar.markdown("---")

# 导航菜单
page = st.sidebar.radio(
    "导航",
    ["🏠 首页", "📈 数据浏览", "📊 统计分析", "🔮 预测分析", 
     "🔬 经济模型", "📉 景气指数", "📄 报告生成"],
    index=0
)

# ==================== 主页面函数 ====================

def show_home():
    """首页"""
    st.title("🏠 宏观经济数据分析平台")
    st.markdown("欢迎访问国家宏观经济数据分析平台！")
    
    # 系统介绍
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("支持指标", f"{len(IndicatorLibrary.NBS_INDICATORS)}+")
    
    with col2:
        stats = managers['data'].get_statistics()
        db_stats = stats.get('database', {})
        st.metric("数据记录", f"{db_stats.get('total_records', 0):,}")
    
    with col3:
        st.metric("分析模型", "6种")
    
    st.markdown("---")
    
    # 功能介绍
    st.subheader("🎯 核心功能")
    
    features = [
        ("数据获取", "支持国家统计局、人民银行等多源数据获取"),
        ("统计分析", "描述性统计、相关性分析、回归分析"),
        ("预测分析", "时间序列预测、趋势外推"),
        ("经济模型", "增长核算、奥肯定律、菲利普斯曲线"),
        ("景气监测", "扩散指数、合成指数计算"),
        ("报告生成", "自动生成Markdown/HTML/PDF报告")
    ]
    
    for i, (title, desc) in enumerate(features):
        col = st.columns(3)[i % 3]
        with col:
            st.markdown(f"**{title}**")
            st.markdown(desc)


def show_data_browser():
    """数据浏览页面"""
    st.title("📈 数据浏览")
    
    # 指标选择
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # 按类别分组
        categories = {}
        for ind in IndicatorLibrary.NBS_INDICATORS:
            if ind.category not in categories:
                categories[ind.category] = []
            categories[ind.category].append(ind)
        
        # 选择类别
        selected_category = st.selectbox(
            "选择指标类别",
            ["全部"] + list(categories.keys())
        )
        
        # 筛选指标
        if selected_category == "全部":
            indicators = IndicatorLibrary.NBS_INDICATORS
        else:
            indicators = categories[selected_category]
        
        # 选择指标
        indicator_options = {f"{ind.code} - {ind.name}": ind.code for ind in indicators}
        selected_display = st.selectbox("选择指标", list(indicator_options.keys()))
        selected_code = indicator_options[selected_display]
    
    with col2:
        st.markdown("### 日期范围")
        start_year = st.number_input("开始年份", 2000, 2030, 2010)
        end_year = st.number_input("结束年份", 2000, 2030, datetime.now().year)
    
    # 获取数据
    if st.button("获取数据", type="primary"):
        with st.spinner("正在获取数据..."):
            ts = managers['data'].fetch(
                selected_code,
                start_date=f"{start_year}-01-01",
                end_date=f"{end_year}-12-31"
            )
            
            if ts:
                st.session_state['current_ts'] = ts
                st.success(f"✓ 成功获取 {len(ts)} 条数据")
            else:
                st.error("❌ 数据获取失败")
    
    # 显示数据
    if 'current_ts' in st.session_state:
        ts = st.session_state['current_ts']
        
        # 指标信息
        with st.expander("📋 指标信息", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("指标名称", ts.name)
                st.metric("单位", ts.meta.indicator.unit)
            with col2:
                st.metric("数据频率", ts.meta.indicator.frequency)
                st.metric("数据点数", len(ts))
            with col3:
                st.metric("起始日期", ts.meta.start_date)
                st.metric("结束日期", ts.meta.end_date)
        
        # 时间序列图
        st.subheader("📊 时间序列图")
        
        fig = managers['visualizer']._plot_time_series_plotly(
            ts, title=f"{ts.name} 变化趋势", show_ma=True, ma_windows=[3, 12]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 数据表格
        with st.expander("📋 数据表格"):
            st.dataframe(ts.data, use_container_width=True)
        
        # 导出按钮
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("导出 Excel"):
                managers['data']._export_data(ts, 'excel', None)
                st.success("导出成功!")
        with col2:
            if st.button("导出 CSV"):
                managers['data']._export_data(ts, 'csv', None)
                st.success("导出成功!")
        with col3:
            if st.button("导出 JSON"):
                managers['data']._export_data(ts, 'json', None)
                st.success("导出成功!")


def show_statistical_analysis():
    """统计分析页面"""
    st.title("📊 统计分析")
    
    if 'current_ts' not in st.session_state:
        st.warning("⚠️ 请先在【数据浏览】页面选择并获取数据")
        return
    
    ts = st.session_state['current_ts']
    
    # 描述性统计
    st.subheader("📈 描述性统计")
    stats = managers['analyzer'].descriptive_stats(ts)
    st.dataframe(stats, use_container_width=True)
    
    # 增长率分析
    st.subheader("📉 增长率分析")
    growth_df = managers['analyzer'].calculate_growth_rates(ts)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("最新同比增长率", f"{growth_df['yoy'].iloc[-1]:.2f}%")
    with col2:
        st.metric("平均同比增长率", f"{growth_df['yoy'].mean():.2f}%")
    
    # 增长率图表
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=growth_df.index, y=growth_df['yoy'],
        mode='lines+markers', name='同比增长率'
    ))
    fig.update_layout(title="同比增长率变化", xaxis_title="时间", yaxis_title="增长率(%)")
    st.plotly_chart(fig, use_container_width=True)
    
    # 平稳性检验
    st.subheader("🔬 平稳性检验 (ADF检验)")
    adf_result = managers['analyzer'].adf_test(ts)
    
    if 'error' not in adf_result:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ADF统计量", f"{adf_result['adf_statistic']:.4f}")
        with col2:
            st.metric("P值", f"{adf_result['p_value']:.4f}")
        with col3:
            st.metric("结论", adf_result['interpretation'])
    else:
        st.error(f"检验失败: {adf_result['error']}")


def show_prediction():
    """预测分析页面"""
    st.title("🔮 预测分析")
    
    if 'current_ts' not in st.session_state:
        st.warning("⚠️ 请先在【数据浏览】页面选择并获取数据")
        return
    
    ts = st.session_state['current_ts']
    
    # 预测参数
    col1, col2 = st.columns([2, 1])
    with col2:
        periods = st.slider("预测期数", 1, 24, 6)
        
        if st.button("开始预测", type="primary"):
            with st.spinner("正在预测..."):
                from core.predictor import Predictor
                predictor = Predictor()
                result = predictor.forecast(ts, periods=periods)
                st.session_state['forecast_result'] = result
    
    # 显示结果
    if 'forecast_result' in st.session_state:
        result = st.session_state['forecast_result']
        
        st.subheader("📊 预测结果")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("预测方法", result.get('method', 'N/A'))
        with col2:
            st.metric("R² 分数", f"{result.get('r_squared', 0):.4f}")
        
        # 预测值表格
        forecast_values = result.get('forecast', [])
        if forecast_values:
            st.subheader("📋 预测值")
            
            forecast_df = pd.DataFrame({
                '期数': range(1, len(forecast_values) + 1),
                '预测值': forecast_values
            })
            st.dataframe(forecast_df, use_container_width=True)
            
            # 预测图表
            fig = go.Figure()
            
            # 历史数据
            fig.add_trace(go.Scatter(
                x=ts.data.index, y=ts.data['value'],
                mode='lines', name='历史数据'
            ))
            
            # 预测数据
            last_date = ts.data.index[-1]
            forecast_dates = pd.date_range(start=last_date, periods=len(forecast_values)+1, freq='M')[1:]
            
            fig.add_trace(go.Scatter(
                x=forecast_dates, y=forecast_values,
                mode='lines+markers', name='预测值',
                line=dict(dash='dash', color='red')
            ))
            
            fig.update_layout(title="预测趋势图", xaxis_title="时间", yaxis_title="数值")
            st.plotly_chart(fig, use_container_width=True)


def show_economic_models():
    """经济模型页面"""
    st.title("🔬 经济模型分析")
    
    # 模型选择
    model_type = st.radio(
        "选择模型",
        ["增长核算模型", "奥肯定律", "菲利普斯曲线"],
        horizontal=True
    )
    
    if model_type == "增长核算模型":
        st.subheader("增长核算模型")
        st.markdown("""
        基于索洛增长模型，将经济增长分解为：
        - 资本贡献
        - 劳动贡献  
        - 全要素生产率（TFP）贡献
        """)
        
        # 参数设置
        capital_share = st.slider("资本产出弹性 (α)", 0.0, 1.0, 0.4, 0.01)
        
        if st.button("运行模型", type="primary"):
            model = GrowthAccountingModel(capital_share=capital_share)
            
            # 获取GDP数据
            gdp_ts = managers['data'].fetch('gdp')
            
            if gdp_ts:
                with st.spinner("正在计算..."):
                    result = model.calculate(gdp_ts)
                
                # 显示结果
                st.subheader("📊 核算结果")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("平均GDP增长率", f"{result['avg_gdp_growth']:.2f}%")
                with col2:
                    st.metric("资本贡献", f"{result['avg_capital_contribution']:.2f}%")
                with col3:
                    st.metric("劳动贡献", f"{result['avg_labor_contribution']:.2f}%")
                with col4:
                    st.metric("TFP贡献", f"{result['avg_tfp_growth']:.2f}%")
                
                # 可视化
                if 'detailed_data' in result:
                    df = result['detailed_data']
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name='资本贡献', x=df.index, y=df['capital_contribution']))
                    fig.add_trace(go.Bar(name='劳动贡献', x=df.index, y=df['labor_contribution']))
                    fig.add_trace(go.Bar(name='TFP贡献', x=df.index, y=df['tfp_growth']))
                    
                    fig.update_layout(
                        barmode='stack',
                        title="经济增长贡献分解",
                        xaxis_title="时间",
                        yaxis_title="贡献百分点"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("❌ 无法获取GDP数据")
    
    elif model_type == "奥肯定律":
        st.subheader("奥肯定律模型")
        st.markdown("""
        分析GDP增长与失业率变动之间的关系。
        
        模型形式：Δu = a + b × g_Y
        
        注意：中国失业率数据与西方国家存在差异，分析结果需谨慎解读。
        """)
        
        st.info("⚠️ 该模型需要失业率数据支持，当前演示模型框架。")
    
    elif model_type == "菲利普斯曲线":
        st.subheader("菲利普斯曲线模型")
        st.markdown("""
        分析失业率与通货膨胀率之间的关系。
        
        模型形式：π = f(u)
        
        注意：中国通胀与失业关系可能呈现不同特征。
        """)
        
        st.info("⚠️ 该模型需要通胀率和失业率数据支持，当前演示模型框架。")


def show_business_cycle():
    """景气指数页面"""
    st.title("📉 经济景气指数")
    
    st.markdown("""
    经济景气指数通过综合多个经济指标，判断经济运行状态：
    - **先行指数**：预测未来经济走势
    - **一致指数**：反映当前经济状态
    - **滞后指数**：确认经济周期转折
    """)
    
    # 计算景气指数
    if st.button("计算景气指数", type="primary"):
        with st.spinner("正在计算..."):
            # 获取指标数据
            leading_codes = [ind.code for ind in IndicatorLibrary.get_leading_indicators()]
            coincident_codes = [ind.code for ind in IndicatorLibrary.get_coincident_indicators()]
            
            leading_data = managers['data'].fetch_multiple(leading_codes[:3])
            coincident_data = managers['data'].fetch_multiple(coincident_codes[:3])
            
            if leading_data and coincident_data:
                # 计算合成指数
                leading_index = managers['analyzer'].calculate_composite_index(leading_data)
                coincident_index = managers['analyzer'].calculate_composite_index(coincident_data)
                
                # 可视化
                fig = go.Figure()
                
                if not leading_index.empty:
                    fig.add_trace(go.Scatter(
                        x=leading_index.index, y=leading_index['composite_index'],
                        mode='lines', name='先行指数', line=dict(color='#E74C3C')
                    ))
                
                if not coincident_index.empty:
                    fig.add_trace(go.Scatter(
                        x=coincident_index.index, y=coincident_index['composite_index'],
                        mode='lines', name='一致指数', line=dict(color='#3498DB')
                    ))
                
                fig.update_layout(
                    title="经济景气指数",
                    xaxis_title="时间",
                    yaxis_title="指数",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("❌ 数据不足，无法计算景气指数")


def show_report_generator():
    """报告生成页面"""
    st.title("📄 报告生成")
    
    if 'current_ts' not in st.session_state:
        st.warning("⚠️ 请先在【数据浏览】页面选择并获取数据")
        return
    
    ts = st.session_state['current_ts']
    
    st.markdown(f"**当前指标**: {ts.name}")
    
    # 报告选项
    col1, col2 = st.columns(2)
    with col1:
        include_forecast = st.checkbox("包含预测分析", value=True)
    with col2:
        forecast_periods = st.slider("预测期数", 3, 24, 12)
    
    # 生成报告
    if st.button("生成报告", type="primary"):
        with st.spinner("正在生成报告..."):
            report_data = managers['reporter'].generate_indicator_report(
                ts,
                include_forecast=include_forecast,
                forecast_periods=forecast_periods
            )
            
            st.success(f"✓ 报告已生成")
            
            # 显示报告内容
            with st.expander("📖 查看报告内容"):
                st.markdown(report_data['content'])
            
            # 下载按钮
            st.download_button(
                label="下载报告",
                data=report_data['content'],
                file_name=f"{ts.meta.indicator.code}_report.md",
                mime="text/markdown"
            )


# ==================== 主程序 ====================

def main():
    """主函数"""
    if page == "🏠 首页":
        show_home()
    elif page == "📈 数据浏览":
        show_data_browser()
    elif page == "📊 统计分析":
        show_statistical_analysis()
    elif page == "🔮 预测分析":
        show_prediction()
    elif page == "🔬 经济模型":
        show_economic_models()
    elif page == "📉 景气指数":
        show_business_cycle()
    elif page == "📄 报告生成":
        show_report_generator()


if __name__ == '__main__':
    main()
