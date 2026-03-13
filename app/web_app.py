#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask Web 服务入口
宏观经济数据分析平台 - Web界面
"""

import os
import sys
from flask import Flask, render_template_string, jsonify, request
import pandas as pd
import numpy as np

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>宏观经济数据分析平台 | Web版</title>
    <style>
        :root {
            --primary-color: #2563eb;
            --primary-dark: #1d4ed8;
            --bg-light: #f8fafc;
            --bg-white: #ffffff;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --border-color: #e2e8f0;
            --radius: 8px;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-light);
            color: var(--text-primary);
            min-height: 100vh;
        }
        
        .header {
            background: var(--bg-white);
            padding: 16px 24px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .header h1 {
            font-size: 20px;
            color: var(--primary-color);
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 24px;
        }
        
        .card {
            background: var(--bg-white);
            border-radius: var(--radius);
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .card h2 {
            font-size: 18px;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border-color);
        }
        
        .data-selector {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-bottom: 24px;
        }
        
        .btn {
            padding: 10px 20px;
            border-radius: var(--radius);
            border: none;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }
        
        .btn-primary {
            background: var(--primary-color);
            color: white;
        }
        
        .btn-primary:hover {
            background: var(--primary-dark);
        }
        
        .btn-outline {
            background: white;
            border: 1px solid var(--primary-color);
            color: var(--primary-color);
        }
        
        .btn-outline.active {
            background: var(--primary-color);
            color: white;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-top: 16px;
        }
        
        .stat-card {
            background: var(--bg-light);
            padding: 16px;
            border-radius: var(--radius);
            text-align: center;
        }
        
        .stat-value {
            font-size: 28px;
            font-weight: 600;
            color: var(--primary-color);
        }
        
        .stat-label {
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 4px;
        }
        
        .chart-container {
            height: 300px;
            margin-top: 16px;
        }
        
        .chart-placeholder {
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--bg-light);
            border-radius: var(--radius);
            color: var(--text-secondary);
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        
        th {
            background: var(--bg-light);
            font-weight: 600;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: var(--text-secondary);
        }
        
        .alert {
            padding: 12px 16px;
            border-radius: var(--radius);
            margin-bottom: 16px;
        }
        
        .alert-success {
            background: #dcfce7;
            color: #166534;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 宏观经济数据分析平台 - Web版</h1>
        <span style="color: var(--text-secondary); font-size: 14px;">数据来源: 国家统计局</span>
    </div>
    
    <div class="container">
        <div class="card">
            <h2>选择分析指标</h2>
            <div class="data-selector">
                <button class="btn btn-outline active" data-type="gdp">GDP 国内生产总值</button>
                <button class="btn btn-outline" data-type="cpi">CPI 居民消费价格指数</button>
                <button class="btn btn-outline" data-type="employment">就业情况</button>
                <button class="btn btn-outline" data-type="trade">对外贸易</button>
            </div>
        </div>
        
        <div class="card">
            <h2>📈 数据概览</h2>
            <div class="stats-grid" id="statsGrid">
                <div class="stat-card">
                    <div class="stat-value">--</div>
                    <div class="stat-label">最新值</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">--</div>
                    <div class="stat-label">同比变化</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">--</div>
                    <div class="stat-label">环比变化</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">--</div>
                    <div class="stat-label">数据年份</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>📊 趋势图表</h2>
            <div class="chart-container">
                <div class="chart-placeholder" id="chartPlaceholder">
                    点击上方按钮加载数据
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>📋 历史数据</h2>
            <table id="dataTable">
                <thead>
                    <tr>
                        <th>年份</th>
                        <th>数值</th>
                        <th>同比变化</th>
                    </tr>
                </thead>
                <tbody id="tableBody">
                    <tr><td colspan="3" style="text-align: center; color: var(--text-secondary);">请选择指标查看数据</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        const dataCache = {};
        
        const dataConfig = {
            gdp: { unit: '万亿元', name: '国内生产总值' },
            cpi: { unit: '%', name: '居民消费价格指数' },
            employment: { unit: '万人', name: '城镇就业人数' },
            trade: { unit: '亿美元', name: '进出口总额' }
        };
        
        const mockData = {
            gdp: {
                years: [2020, 2021, 2022, 2023, 2024],
                values: [101.6, 110.4, 121.0, 121.0, 126.1],
                changes: [2.2, 8.4, 9.5, 3.0, 4.2]
            },
            cpi: {
                years: [2020, 2021, 2022, 2023, 2024],
                values: [102.5, 100.9, 102.9, 102.0, 101.5],
                changes: [2.5, 0.9, 2.9, 2.0, 1.5]
            },
            employment: {
                years: [2020, 2021, 2022, 2023, 2024],
                values: [750.6, 783.5, 801.9, 813.1, 825.0],
                changes: [2.1, 4.4, 2.3, 1.4, 1.5]
            },
            trade: {
                years: [2020, 2021, 2022, 2023, 2024],
                values: [46500, 59000, 63000, 59000, 62000],
                changes: [1.9, 26.9, 6.8, -6.3, 5.1]
            }
        };
        
        document.querySelectorAll('.btn[data-type]').forEach(btn => {
            btn.addEventListener('click', function() {
                document.querySelectorAll('.btn[data-type]').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                loadData(this.dataset.type);
            });
        });
        
        function loadData(type) {
            const data = mockData[type];
            const config = dataConfig[type];
            
            const latest = data.values[data.values.length - 1];
            const prev = data.values[data.values.length - 2];
            const yoyChange = data.changes[data.changes.length - 1];
            const momChange = ((latest - prev) / prev * 100).toFixed(1);
            
            document.getElementById('statsGrid').innerHTML = `
                <div class="stat-card">
                    <div class="stat-value">${latest} ${config.unit}</div>
                    <div class="stat-label">最新值</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${yoyChange > 0 ? '+' : ''}${yoyChange}%</div>
                    <div class="stat-label">同比变化</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${momChange > 0 ? '+' : ''}${momChange}%</div>
                    <div class="stat-label">环比变化</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.years.length} 年</div>
                    <div class="stat-label">数据年份</div>
                </div>
            `;
            
            document.getElementById('chartPlaceholder').innerHTML = `
                <div style="width: 100%; height: 100%;">
                    <canvas id="chartCanvas" style="width: 100%; height: 100%;"></canvas>
                </div>
            `;
            
            drawChart(data);
            
            let tbody = '';
            for (let i = data.years.length - 1; i >= 0; i--) {
                tbody += `<tr>
                    <td>${data.years[i]}</td>
                    <td>${data.values[i]} ${config.unit}</td>
                    <td style="color: ${data.changes[i] >= 0 ? '#16a34a' : '#dc2626'}">
                        ${data.changes[i] > 0 ? '+' : ''}${data.changes[i]}%
                    </td>
                </tr>`;
            }
            document.getElementById('tableBody').innerHTML = tbody;
        }
        
        function drawChart(data) {
            const canvas = document.getElementById('chartCanvas');
            if (!canvas) return;
            
            const ctx = canvas.getContext('2d');
            const width = canvas.offsetWidth;
            const height = canvas.offsetHeight;
            canvas.width = width;
            canvas.height = height;
            
            const padding = 40;
            const chartWidth = width - padding * 2;
            const chartHeight = height - padding * 2;
            
            const maxVal = Math.max(...data.values) * 1.1;
            const minVal = Math.min(...data.values) * 0.9;
            
            ctx.clearRect(0, 0, width, height);
            
            ctx.strokeStyle = '#e2e8f0';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 4; i++) {
                const y = padding + (chartHeight / 4) * i;
                ctx.beginPath();
                ctx.moveTo(padding, y);
                ctx.lineTo(width - padding, y);
                ctx.stroke();
            }
            
            ctx.strokeStyle = '#2563eb';
            ctx.lineWidth = 2;
            ctx.beginPath();
            data.values.forEach((val, i) => {
                const x = padding + (chartWidth / (data.values.length - 1)) * i;
                const y = padding + chartHeight - ((val - minVal) / (maxVal - minVal)) * chartHeight;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
            
            ctx.fillStyle = '#2563eb';
            data.values.forEach((val, i) => {
                const x = padding + (chartWidth / (data.values.length - 1)) * i;
                const y = padding + chartHeight - ((val - minVal) / (maxVal - minVal)) * chartHeight;
                ctx.beginPath();
                ctx.arc(x, y, 4, 0, Math.PI * 2);
                ctx.fill();
            });
            
            ctx.fillStyle = '#64748b';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';
            data.years.forEach((year, i) => {
                const x = padding + (chartWidth / (data.values.length - 1)) * i;
                ctx.fillText(year, x, height - 10);
            });
        }
        
        loadData('gdp');
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/data/<data_type>')
def get_data(data_type):
    data_map = {
        'gdp': {
            'years': [2020, 2021, 2022, 2023, 2024],
            'values': [101.6, 110.4, 121.0, 121.0, 126.1],
            'changes': [2.2, 8.4, 9.5, 3.0, 4.2]
        },
        'cpi': {
            'years': [2020, 2021, 2022, 2023, 2024],
            'values': [102.5, 100.9, 102.9, 102.0, 101.5],
            'changes': [2.5, 0.9, 2.9, 2.0, 1.5]
        },
        'employment': {
            'years': [2020, 2021, 2022, 2023, 2024],
            'values': [750.6, 783.5, 801.9, 813.1, 825.0],
            'changes': [2.1, 4.4, 2.3, 1.4, 1.5]
        }
    }
    
    if data_type in data_map:
        return jsonify({'success': True, 'data': data_map[data_type]})
    return jsonify({'success': False, 'message': 'Invalid data type'})

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    data_type = data.get('type', 'gdp')
    
    result = {
        'type': data_type,
        'trend': '稳中有升' if data_type == 'gdp' else '温和上涨',
        'prediction': '2025年预计增长4.5%'
    }
    
    return jsonify({'success': True, 'result': result})

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'service': 'national_stats'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🌐 启动宏观经济数据分析平台 Web服务...")
    print(f"   访问地址: http://localhost:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)
