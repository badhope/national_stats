#!/usr/bin/env python
"""
增强版Web应用 - Streamlit仪表盘
集成所有高级功能：大数据处理、预测、拟合等
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path
import json
import time

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, IndicatorLibrary, IndicatorCategory
from core import DataManager, StatisticalAnalyzer, Visualizer, ReportGenerator
from core.models import GrowthAccountingModel, OkunLawModel, PhillipsCurveModel
from core.predictor import Predictor
from core.fitter import AdvancedFitter, fit_trend_analysis
from core.big_data_processor import BigDataProcessor


# ==================== 页面配置 ====================

st.set_page_config(
    page_title="宏观经济智能分析平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 样式设置 ====================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 全局初始化 ====================

@st.cache_resource
def init_managers():
    """初始化管理器（缓存）"""
    return {
        'data': DataManager(),
        'analyzer': StatisticalAnalyzer(),
        'visualizer': Visualizer(),
        'reporter': ReportGenerator(),
        'big_data': BigDataProcessor(use_dask=True, use_ray=True)
    }

managers = init_managers()

# ==================== 侧边栏 ====================

with st.sidebar:
    st.header("📈 宏观经济智能分析平台")
    st.markdown("---")
    
    # 主要功能选择
    app_mode = st.selectbox(
        "选择功能模块",
        [
            "📊 数据概览",
            "🔍 指标分析", 
            "🔮 智能预测",
            "🎯 数据拟合",
            "🔄 经济模型",
            "🌐 大数据洞察",
            "📋 报告生成"
        ]
    )
    
    st.markdown("---")
    
    # 全局参数设置
    st.subheader("⚙️ 全局设置")
    date_range = st.date_input(
        "数据时间范围",
        value=[datetime(2020, 1, 1), datetime.now()],
        key="global_date_range"
    )
    
    if len(date_range) == 2:
        start_date = date_range[0].strftime("%Y-%m")
        end_date = date_range[1].strftime("%Y-%m")
    else:
        start_date = "2020-01"
        end_date = datetime.now().strftime("%Y-%m")

# ==================== 主页面内容 ====================

if app_mode == "📊 数据概览":
    st.markdown('<div class="main-header">宏观经济数据概览</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("核心指标数量", len(IndicatorLibrary.NBS_INDICATORS))
    
    with col2:
        st.metric("领先指标", len(IndicatorLibrary.get_leading_indicators()))
    
    with col3:
        st.metric("同步指标", len(IndicatorLibrary.get_coincident_indicators()))
    
    with col4:
        st.metric("滞后指标", len(IndicatorLibrary.get_lagging_indicators()))
    
    # 指标分类展示
    st.markdown('<div class="section-header">指标分类概览</div>', unsafe_allow_html=True)
    
    categories = {}
    for ind in IndicatorLibrary.NBS_INDICATORS:
        if ind.category not in categories:
            categories[ind.category] = []
        categories[ind.category].append(ind)
    
    cols = st.columns(min(4, len(categories)))
    for i, (category, indicators) in enumerate(categories.items()):
        with cols[i % len(cols)]:
            st.markdown(f"**{category.upper()}**")
            st.metric(f"指标数量", len(indicators))
            st.write(", ".join([ind.code for ind in indicators[:3]]))

elif app_mode == "🔍 指标分析":
    st.markdown('<div class="main-header">指标深度分析</div>', unsafe_allow_html=True)
    
    # 选择指标
    indicator_options = [f"{ind.code} - {ind.name}" for ind in IndicatorLibrary.NBS_INDICATORS]
    selected_indicator = st.selectbox("选择要分析的指标", indicator_options)
    indicator_code = selected_indicator.split(" - ")[0]
    
    if indicator_code:
        with st.spinner("正在获取数据..."):
            ts = managers['data'].fetch(indicator_code, start_date, end_date)
        
        if ts is not None:
            # 基本信息卡片
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("基本信息")
                st.write(f"**指标名称:** {ts.meta.indicator_name}")
                st.write(f"**数据频率:** {ts.meta.frequency}")
                st.write(f"**单位:** {ts.meta.unit}")
                st.write(f"**数据源:** {ts.meta.source}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("数据概况")
                st.write(f"**数据点数:** {len(ts)}")
                st.write(f"**时间跨度:** {ts.meta.start_date} 至 {ts.meta.end_date}")
                if len(ts) > 0:
                    st.write(f"**最新值:** {ts.data['value'].iloc[-1]:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("统计特征")
                if len(ts) > 1:
                    growth_rate = ((ts.data['value'].iloc[-1] / ts.data['value'].iloc[-2]) - 1) * 100
                    st.write(f"**最新环比:** {growth_rate:.2f}%")
                st.write(f"**均值:** {ts.data['value'].mean():.2f}")
                st.write(f"**标准差:** {ts.data['value'].std():.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 图表展示
            st.markdown('<div class="section-header">数据可视化</div>', unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs(["时间序列图", "统计分析", "增长分析"])
            
            with tab1:
                fig = managers['visualizer'].plot_time_series(ts)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # 描述性统计
                desc_stats = managers['analyzer'].descriptive_stats(ts)
                st.write("**描述性统计:**")
                st.json(desc_stats)
                
                # 平稳性检验
                try:
                    adf_result = managers['analyzer'].adf_test(ts)
                    st.write("**平稳性检验 (ADF):**")
                    st.write(f"统计量: {adf_result.get('statistic', 'N/A'):.4f}")
                    st.write(f"P值: {adf_result.get('p_value', 'N/A'):.4f}")
                    st.write(f"结论: {adf_result.get('interpretation', 'N/A')}")
                except Exception as e:
                    st.warning(f"平稳性检验失败: {e}")
            
            with tab3:
                try:
                    growth_df = managers['analyzer'].calculate_growth_rates(ts)
                    if len(growth_df) > 0:
                        fig_growth = go.Figure()
                        fig_growth.add_trace(go.Scatter(
                            x=growth_df.index,
                            y=growth_df['yoy'],
                            mode='lines+markers',
                            name='同比增速 (%)'
                        ))
                        fig_growth.update_layout(
                            title="同比增长率趋势",
                            xaxis_title="时间",
                            yaxis_title="增长率 (%)"
                        )
                        st.plotly_chart(fig_growth, use_container_width=True)
                except Exception as e:
                    st.warning(f"增长分析失败: {e}")

elif app_mode == "🔮 智能预测":
    st.markdown('<div class="main-header">智能预测分析</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        indicator_options = [f"{ind.code} - {ind.name}" for ind in IndicatorLibrary.NBS_INDICATORS]
        selected_indicator = st.selectbox("选择预测指标", indicator_options)
        indicator_code = selected_indicator.split(" - ")[0]
    
    with col2:
        forecast_periods = st.number_input("预测期数", min_value=1, max_value=60, value=12)
        method = st.selectbox("预测方法", ["auto", "arima", "prophet", "xgboost", "ensemble"])
    
    if indicator_code and st.button("开始预测", type="primary"):
        with st.spinner("正在执行预测分析..."):
            ts = managers['data'].fetch(indicator_code, start_date, end_date)
            
            if ts is not None:
                try:
                    predictor = Predictor(method=method)
                    result = predictor.forecast(ts, periods=forecast_periods)
                    
                    # 展示预测结果
                    st.success(f"✅ 预测完成！使用方法: {result['method']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("预测期数", result['periods'])
                        if 'r_squared' in result.get('training_info', {}):
                            st.metric("训练R²", f"{result['training_info']['r_squared']:.4f}")
                    
                    with col2:
                        if 'evaluation' in result:
                            eval_metrics = result['evaluation']
                            st.metric("测试R²", f"{eval_metrics.get('r2', 0):.4f}")
                            st.metric("RMSE", f"{eval_metrics.get('rmse', 0):.4f}")
                    
                    # 预测图表
                    st.markdown('<div class="section-header">预测结果可视化</div>', unsafe_allow_html=True)
                    
                    # 合并历史数据和预测数据
                    historical_x = list(range(len(ts.data)))
                    historical_y = ts.data['value'].tolist()
                    forecast_x = list(range(len(ts.data), len(ts.data) + forecast_periods))
                    forecast_y = result['forecast']
                    
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(
                        x=historical_x,
                        y=historical_y,
                        mode='lines',
                        name='历史数据',
                        line=dict(color='blue')
                    ))
                    fig_pred.add_trace(go.Scatter(
                        x=forecast_x,
                        y=forecast_y,
                        mode='lines+markers',
                        name='预测数据',
                        line=dict(color='red', dash='dash')
                    ))
                    fig_pred.update_layout(
                        title=f"{ts.meta.indicator_name} 预测分析",
                        xaxis_title="时间",
                        yaxis_title=ts.meta.unit
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # 预测值表格
                    st.markdown('<div class="section-header">详细预测值</div>', unsafe_allow_html=True)
                    pred_df = pd.DataFrame({
                        '预测日期': result.get('forecast_dates', [])[:forecast_periods],
                        '预测值': [f"{val:.2f}" for val in forecast_y]
                    })
                    st.dataframe(pred_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"预测失败: {str(e)}")

elif app_mode == "🎯 数据拟合":
    st.markdown('<div class="main-header">高级数据拟合</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        indicator_options = [f"{ind.code} - {ind.name}" for ind in IndicatorLibrary.NBS_INDICATORS]
        selected_indicator = st.selectbox("选择拟合指标", indicator_options)
        indicator_code = selected_indicator.split(" - ")[0]
    
    with col2:
        extrapolate_periods = st.number_input("外推期数", min_value=0, max_value=24, value=6)
        methods = st.multiselect("拟合方法", 
                               ["polynomial", "exponential", "logistic", "spline"],
                               default=["polynomial", "exponential"])
    
    if indicator_code and st.button("开始拟合分析", type="primary"):
        with st.spinner("正在进行数据拟合..."):
            ts = managers['data'].fetch(indicator_code, start_date, end_date)
            
            if ts is not None:
                try:
                    # 准备数据
                    x = np.arange(len(ts.data))
                    y = ts.data['value'].values
                    
                    # 执行拟合
                    fitter = AdvancedFitter()
                    result = fitter.fit(x, y, methods=methods)
                    
                    # 展示拟合结果
                    st.success(f"✅ 拟合完成！最佳方法: {result['best_method']}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("最佳R²", f"{result['r_squared']:.4f}")
                    with col2:
                        st.metric("拟合方法数", len(result['all_results']))
                    with col3:
                        if 'trend_direction' in result:
                            st.metric("趋势方向", result['trend_direction'])
                    
                    # 拟合图表
                    st.markdown('<div class="section-header">拟合结果可视化</div>', unsafe_allow_html=True)
                    
                    fig_fit = go.Figure()
                    fig_fit.add_trace(go.Scatter(
                        x=x,
                        y=y,
                        mode='lines+markers',
                        name='原始数据',
                        line=dict(color='blue')
                    ))
                    
                    # 添加拟合曲线
                    x_smooth = np.linspace(0, len(x)-1, 200)
                    y_fitted = fitter.predict(x_smooth)
                    fig_fit.add_trace(go.Scatter(
                        x=x_smooth,
                        y=y_fitted,
                        mode='lines',
                        name=f'拟合曲线 ({result["best_method"]})',
                        line=dict(color='red', width=3)
                    ))
                    
                    fig_fit.update_layout(
                        title=f"{ts.meta.indicator_name} 数据拟合分析",
                        xaxis_title="时间索引",
                        yaxis_title=ts.meta.unit
                    )
                    st.plotly_chart(fig_fit, use_container_width=True)
                    
                    # 参数详情
                    if 'parameters' in result:
                        st.markdown('<div class="section-header">拟合参数</div>', unsafe_allow_html=True)
                        param_df = pd.DataFrame([
                            {"参数": k, "值": f"{v:.6f}"} 
                            for k, v in result['parameters'].items()
                        ])
                        st.dataframe(param_df, use_container_width=True)
                    
                    # 外推预测
                    if extrapolate_periods > 0:
                        st.markdown('<div class="section-header">外推预测</div>', unsafe_allow_html=True)
                        x_new = np.arange(len(x), len(x) + extrapolate_periods)
                        extrapolation = fitter.extrapolate(x_new)
                        
                        pred_df = pd.DataFrame({
                            '期数': range(1, extrapolate_periods + 1),
                            '预测值': [f"{val:.2f}" for val in extrapolation['predicted_values']],
                            '置信下限': [f"{val:.2f}" for val in extrapolation['confidence_lower']],
                            '置信上限': [f"{val:.2f}" for val in extrapolation['confidence_upper']]
                        })
                        st.dataframe(pred_df, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"拟合失败: {str(e)}")

elif app_mode == "🔄 经济模型":
    st.markdown('<div class="main-header">经济计量模型</div>', unsafe_allow_html=True)
    
    model_type = st.selectbox("选择经济模型", 
                             ["增长核算模型", "奥肯定律", "菲利普斯曲线"])
    
    if model_type == "增长核算模型":
        st.markdown('<div class="section-header">增长核算分析</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            gdp_indicator = st.selectbox("GDP指标", 
                                       [f"{ind.code} - {ind.name}" for ind in IndicatorLibrary.NBS_INDICATORS 
                                        if 'gdp' in ind.code.lower()])
        with col2:
            labor_indicator = st.selectbox("劳动投入指标", 
                                         [f"{ind.code} - {ind.name}" for ind in IndicatorLibrary.NBS_INDICATORS 
                                          if 'labor' in ind.code.lower() or '就业' in ind.name])
        with col3:
            capital_share = st.number_input("资本份额", min_value=0.1, max_value=0.9, value=0.4, step=0.05)
        
        if st.button("运行增长核算", type="primary"):
            with st.spinner("正在计算增长核算..."):
                try:
                    gdp_code = gdp_indicator.split(" - ")[0]
                    labor_code = labor_indicator.split(" - ")[0]
                    
                    gdp_ts = managers['data'].fetch(gdp_code, start_date, end_date)
                    labor_ts = managers['data'].fetch(labor_code, start_date, end_date)
                    # 简化处理：假设资本存量数据
                    capital_ts = gdp_ts  # 临时使用GDP作为代理
                    
                    if all([gdp_ts, labor_ts, capital_ts]):
                        model = GrowthAccountingModel(capital_share=capital_share)
                        result = model.calculate(gdp_ts, labor_ts, capital_ts)
                        
                        # 展示结果
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("GDP增长率", f"{result['gdp_growth']:.2f}%")
                        with col2:
                            st.metric("劳动贡献", f"{result['labor_contribution']:.2f}%")
                        with col3:
                            st.metric("资本贡献", f"{result['capital_contribution']:.2f}%")
                        with col4:
                            st.metric("TFP增长", f"{result['tfp_growth']:.2f}%")
                        
                        # 贡献分解图
                        fig_contrib = go.Figure(data=[
                            go.Bar(name='劳动贡献', x=['贡献率'], y=[result['labor_contribution']], marker_color='blue'),
                            go.Bar(name='资本贡献', x=['贡献率'], y=[result['capital_contribution']], marker_color='red'),
                            go.Bar(name='TFP贡献', x=['贡献率'], y=[result['tfp_growth']], marker_color='green')
                        ])
                        fig_contrib.update_layout(
                            title="增长源泉分解",
                            yaxis_title="贡献率 (%)",
                            barmode='stack'
                        )
                        st.plotly_chart(fig_contrib, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"增长核算失败: {str(e)}")

elif app_mode == "🌐 大数据洞察":
    st.markdown('<div class="main-header">大数据智能洞察</div>', unsafe_allow_html=True)
    
    st.info("💡 此功能可同时分析多个宏观经济指标，发现隐藏的关联模式和趋势")
    
    # 选择多个指标
    all_indicators = [f"{ind.code} - {ind.name}" for ind in IndicatorLibrary.NBS_INDICATORS]
    selected_indicators = st.multiselect(
        "选择分析指标（建议3-10个）",
        all_indicators,
        default=all_indicators[:5]
    )
    
    if len(selected_indicators) >= 2:
        operations = st.multiselect(
            "选择分析操作",
            ["相关性分析", "聚类分析", "批量预测", "异常检测"],
            default=["相关性分析", "聚类分析"]
        )
        
        if st.button("启动大数据分析", type="primary"):
            with st.spinner("正在进行大规模数据分析..."):
                try:
                    # 获取数据
                    indicator_codes = [ind.split(" - ")[0] for ind in selected_indicators]
                    batch_result = managers['big_data'].batch_process_indicators(
                        indicator_codes, start_date, end_date
                    )
                    
                    if batch_result['successful_data']:
                        # 执行大数据分析
                        operations_map = {
                            "相关性分析": "correlation",
                            "聚类分析": "clustering", 
                            "批量预测": "forecasting",
                            "异常检测": "anomaly_detection"
                        }
                        selected_ops = [operations_map[op] for op in operations]
                        
                        analysis_result = managers['big_data'].process_large_dataset(
                            batch_result['successful_data'], selected_ops
                        )
                        
                        st.success(f"✅ 大数据分析完成！处理时间: {analysis_result['processing_time']:.2f}秒")
                        
                        # 展示结果
                        if 'correlation' in analysis_result and analysis_result['correlation']['success']:
                            st.markdown('<div class="section-header">强相关关系发现</div>', unsafe_allow_html=True)
                            corr_pairs = analysis_result['correlation']['strongest_pairs'][:10]
                            for pair in corr_pairs:
                                st.write(f"📊 {pair['indicator1']} ↔ {pair['indicator2']}: "
                                       f"{pair['correlation']:.3f} ({pair['direction']})")
                        
                        if 'clustering' in analysis_result and analysis_result['clustering']['success']:
                            st.markdown('<div class="section-header">指标聚类分析</div>', unsafe_allow_html=True)
                            clusters = analysis_result['clustering']['clusters']
                            for cluster_id, indicators in clusters.items():
                                st.write(f"蔟 {cluster_id + 1}: {', '.join(indicators)}")
                        
                        if 'forecasting' in analysis_result and analysis_result['forecasting']['success']:
                            st.markdown('<div class="section-header">批量预测摘要</div>', unsafe_allow_html=True)
                            forecasts = analysis_result['forecasting']['forecasts']
                            successful = analysis_result['forecasting']['successful_predictions']
                            st.write(f"成功预测指标: {successful}/{len(forecasts)}")
                        
                except Exception as e:
                    st.error(f"大数据分析失败: {str(e)}")

elif app_mode == "📋 报告生成":
    st.markdown('<div class="main-header">智能报告生成</div>', unsafe_allow_html=True)
    
    report_type = st.selectbox("报告类型", ["单指标分析报告", "多指标对比报告", "经济形势分析报告"])
    
    if report_type == "单指标分析报告":
        indicator_options = [f"{ind.code} - {ind.name}" for ind in IndicatorLibrary.NBS_INDICATORS]
        selected_indicator = st.selectbox("选择指标", indicator_options)
        indicator_code = selected_indicator.split(" - ")[0]
        
        if indicator_code and st.button("生成报告", type="primary"):
            with st.spinner("正在生成分析报告..."):
                try:
                    ts = managers['data'].fetch(indicator_code, start_date, end_date)
                    if ts is not None:
                        report_data = managers['reporter'].generate_indicator_report(ts)
                        st.success(f"✅ 报告生成完成！")
                        st.download_button(
                            label="📥 下载报告",
                            data=json.dumps(report_data, ensure_ascii=False, indent=2),
                            file_name=f"{indicator_code}_report.json",
                            mime="application/json"
                        )
                except Exception as e:
                    st.error(f"报告生成失败: {str(e)}")

# ==================== 页脚 ====================

st.markdown("---")
st.caption("📊 宏观经济智能分析平台 | 数据来源：国家统计局等官方机构 | 更新时间：" + 
          datetime.now().strftime("%Y-%m-%d %H:%M:%S"))