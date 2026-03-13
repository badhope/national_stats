#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit 交互式数据仪表板
宏观经济数据分析平台 - 可视化界面
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="宏观经济数据分析平台",
    page_icon="📊",
    layout="wide"
)

MOCK_DATA = {
    'gdp': {
        'title': '国内生产总值 (GDP)',
        'years': list(range(2015, 2025)),
        'values': [68.9, 74.0, 83.2, 87.4, 91.9, 98.6, 101.6, 110.4, 121.0, 126.1],
        'unit': '万亿元',
        'description': '国内生产总值是指在一定时期内，一个国家或地区的经济中所生产出的全部最终产品和劳务的价值总和。'
    },
    'cpi': {
        'title': '居民消费价格指数 (CPI)',
        'years': list(range(2015, 2025)),
        'values': [101.4, 102.0, 101.6, 101.9, 102.1, 101.5, 102.5, 100.9, 102.9, 101.5],
        'unit': '上年=100',
        'description': '居民消费价格指数是反映居民家庭一般所购买的消费品和服务项目价格水平变动情况的宏观经济指标。'
    },
    'employment': {
        'title': '城镇就业人数',
        'years': list(range(2015, 2025)),
        'values': [414.3, 434.3, 454.9, 474.2, 495.3, 514.1, 542.3, 587.3, 613.1, 625.0],
        'unit': '万人',
        'description': '城镇就业人员是指在城镇地区从事一定社会劳动并取得劳动报酬或者经营收入的劳动力。'
    },
    'ppi': {
        'title': '工业生产者出厂价格指数 (PPI)',
        'years': list(range(2015, 2025)),
        'values': [98.1, 98.6, 99.8, 101.7, 105.2, 100.8, 106.3, 108.1, 104.1, 102.8],
        'unit': '上年=100',
        'description': '工业生产者出厂价格指数是反映一定时期内工业生产所需购进物品的价格变动趋势和程度的统计指标。'
    }
}

def calculate_growth_rate(values):
    rates = []
    for i in range(1, len(values)):
        rate = (values[i] - values[i-1]) / values[i-1] * 100
        rates.append(round(rate, 1))
    return rates

def create_line_chart(data, title, unit):
    df = pd.DataFrame({
        '年份': data['years'],
        '数值': data['values']
    })
    
    fig = px.line(df, x='年份', y='数值', title=f'{title}趋势图',
                  markers=True, template='plotly_white')
    fig.update_layout(yaxis_title=f'数值 ({unit})')
    return fig

def create_bar_chart(data, title, unit):
    rates = calculate_growth_rate(data['values'])
    years = data['years'][1:]
    
    colors = ['#16a34a' if r >= 0 else '#dc2626' for r in rates]
    
    fig = go.Figure(data=[
        go.Bar(x=years, y=rates, marker_color=colors)
    ])
    fig.update_layout(title=f'{title}同比增长率(%)',
                      template='plotly_white',
                      yaxis_title='同比增长率 (%)')
    return fig

def main():
    st.title("📊 宏观经济数据分析平台")
    st.markdown("基于国家统计局官方数据的专业宏观经济分析工具")
    
    with st.sidebar:
        st.header("⚙️ 设置")
        data_type = st.selectbox(
            "选择分析指标",
            options=list(MOCK_DATA.keys()),
            format_func=lambda x: MOCK_DATA[x]['title']
        )
        
        st.divider()
        
        st.header("📈 图表类型")
        chart_type = st.radio(
            "选择可视化方式",
            options=["折线图", "柱状图", "数据表格"],
            horizontal=True
        )
        
        st.divider()
        
        st.header("ℹ️ 关于")
        st.info("""
        **数据来源**: 国家统计局
        
        **更新周期**: 年度数据
        
        **指标说明**: 
        - GDP: 国内生产总值
        - CPI: 居民消费价格指数
        - 就业: 城镇就业人数
        - PPI: 工业生产者出厂价格指数
        """)
    
    data = MOCK_DATA[data_type]
    
    col1, col2, col3, col4 = st.columns(4)
    
    latest_value = data['values'][-1]
    prev_value = data['values'][-2]
    yoy_change = (latest_value - prev_value) / prev_value * 100
    
    with col1:
        st.metric(
            label=f"最新{data['unit']}",
            value=f"{latest_value:,.1f}",
            delta=f"{yoy_change:+.1f}%"
        )
    
    with col2:
        avg_value = np.mean(data['values'])
        st.metric(label="历史均值", value=f"{avg_value:,.1f}")
    
    with col3:
        max_value = max(data['values'])
        max_year = data['values'].index(max_value) + data['years'][0]
        st.metric(label="历史最大值", value=f"{max_value:,.1f}", delta=f"{max_year}年")
    
    with col4:
        min_value = min(data['values'])
        min_year = data['values'].index(min_value) + data['years'][0]
        st.metric(label="历史最小值", value=f"{min_value:,.1f}", delta=f"{min_year}年")
    
    st.divider()
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader(f"📈 {data['title']}趋势")
        
        if chart_type == "折线图":
            fig = create_line_chart(data, data['title'], data['unit'])
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "柱状图":
            fig = create_bar_chart(data, data['title'], data['unit'])
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            df = pd.DataFrame({
                '年份': data['years'],
                f'数值 ({data["unit"]})': data['values']
            })
            df['同比增长率(%)'] = [''] + calculate_growth_rate(data['values'])
            st.dataframe(df, use_container_width=True)
    
    with col_right:
        st.subheader("📋 指标说明")
        st.write(data['description'])
        
        st.subheader("📊 统计摘要")
        stats = {
            '指标': ['数据年份', '最新值', '年均增长率', '最大年', '最小年'],
            '数值': [
                f"{len(data['years'])}年",
                f"{latest_value:,.1f}",
                f"{np.mean(calculate_growth_rate(data['values'])):+.1f}%",
                f"{data['years'][data['values'].index(max(data['values']))]}年",
                f"{data['years'][data['values'].index(min(data['values']))]}年"
            ]
        }
        st.table(pd.DataFrame(stats))
    
    st.divider()
    
    st.subheader("📊 多指标对比")
    
    compare_metrics = st.multiselect(
        "选择要对比的指标",
        options=list(MOCK_DATA.keys()),
        default=[data_type]
    )
    
    if compare_metrics:
        fig = go.Figure()
        for metric in compare_metrics:
            m_data = MOCK_DATA[metric]
            normalized = [v / m_data['values'][0] * 100 for v in m_data['values']]
            fig.add_trace(go.Scatter(
                x=m_data['years'],
                y=normalized,
                mode='lines+markers',
                name=m_data['title']
            ))
        
        fig.update_layout(
            title="多指标对比 (以2015年为基期=100)",
            xaxis_title="年份",
            yaxis_title="指数",
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
