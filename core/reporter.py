"""
报告生成模块
自动生成宏观经济分析报告，支持Markdown和PDF格式
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
import logging
import base64

# 导入基础模块
import sys
sys.path.append('..')
from config import Config, PathConfig
from models.time_series import MacroTimeSeries
from core.analyzer import StatisticalAnalyzer
from core.visualizer import Visualizer


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self):
        """初始化报告生成器"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.analyzer = StatisticalAnalyzer()
        self.visualizer = Visualizer()
        self.report_dir = PathConfig.EXPORT_DIR
    
    def generate_indicator_report(self, 
                                 ts: MacroTimeSeries,
                                 include_forecast: bool = True,
                                 forecast_periods: int = 12,
                                 output_format: str = 'markdown') -> Dict[str, Any]:
        """
        生成单指标深度分析报告
        
        Args:
            ts: 时间序列对象
            include_forecast: 是否包含预测
            forecast_periods: 预测期数
            output_format: 输出格式
        
        Returns:
            包含报告内容和文件路径的字典
        """
        report_data = {
            'indicator_code': ts.meta.indicator.code,
            'indicator_name': ts.name,
            'generated_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'period': f"{ts.meta.start_date} 至 {ts.meta.end_date}",
            'data_points': len(ts)
        }
        
        # 1. 描述性统计
        report_data['descriptive_stats'] = self.analyzer.descriptive_stats(ts).to_dict()
        
        # 2. 增长率分析
        growth_df = self.analyzer.calculate_growth_rates(ts)
        report_data['growth_analysis'] = {
            'latest_yoy': growth_df['yoy'].iloc[-1] if not growth_df['yoy'].empty else None,
            'avg_yoy': growth_df['yoy'].mean(),
            'yoy_trend': '上升' if growth_df['yoy'].iloc[-1] > growth_df['yoy'].mean() else '下降'
        }
        
        # 3. 平稳性检验
        adf_result = self.analyzer.adf_test(ts)
        report_data['stationarity'] = adf_result
        
        # 4. 生成图表
        chart_paths = self._generate_charts(ts, growth_df)
        report_data['charts'] = chart_paths
        
        # 5. 预测分析
        if include_forecast:
            forecast_result = self._generate_forecast(ts, forecast_periods)
            report_data['forecast'] = forecast_result
        
        # 6. 生成报告文档
        if output_format == 'markdown':
            content = self._render_markdown_report(report_data)
            file_path = self._save_markdown(content, ts.meta.indicator.code)
        elif output_format == 'pdf':
            content = self._render_markdown_report(report_data)
            file_path = self._save_pdf(content, ts.meta.indicator.code, chart_paths)
        else:
            content = self._render_html_report(report_data)
            file_path = self._save_html(content, ts.meta.indicator.code)
        
        report_data['content'] = content
        report_data['file_path'] = str(file_path)
        
        return report_data
    
    def generate_comparison_report(self,
                                  ts_dict: Dict[str, MacroTimeSeries],
                                  output_format: str = 'markdown') -> Dict[str, Any]:
        """
        生成多指标对比分析报告
        
        Args:
            ts_dict: 指标字典
            output_format: 输出格式
        
        Returns:
            报告数据字典
        """
        report_data = {
            'generated_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'indicator_count': len(ts_dict),
            'indicators': [ts.name for ts in ts_dict.values()]
        }
        
        # 1. 相关性分析
        corr_matrix = self.analyzer.correlation_matrix(ts_dict)
        report_data['correlation_matrix'] = corr_matrix.to_dict()
        
        # 2. 相关性热力图
        heatmap_path = self.report_dir / 'charts' / f"correlation_heatmap_{datetime.now().strftime('%Y%m%d')}.png"
        self.visualizer.plot_correlation_heatmap(corr_matrix, save_path=heatmap_path)
        report_data['heatmap_path'] = str(heatmap_path)
        
        # 3. 多指标对比图
        comparison_path = self.report_dir / 'charts' / f"comparison_{datetime.now().strftime('%Y%m%d')}.png"
        self.visualizer.plot_multiple_series(ts_dict, normalize=True, save_path=comparison_path)
        report_data['comparison_chart'] = str(comparison_path)
        
        # 4. 生成报告
        content = self._render_comparison_markdown(report_data)
        file_path = self._save_markdown(content, 'comparison_report')
        
        report_data['content'] = content
        report_data['file_path'] = str(file_path)
        
        return report_data
    
    def generate_economic_model_report(self,
                                      model_type: str,
                                      model_result: Dict[str, Any],
                                      ts_dict: Dict[str, MacroTimeSeries]) -> Dict[str, Any]:
        """
        生成经济模型分析报告
        
        Args:
            model_type: 模型类型
            model_result: 模型运行结果
            ts_dict: 相关时间序列
        
        Returns:
            报告数据字典
        """
        report_data = {
            'generated_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': model_type,
            'model_result': model_result
        }
        
        # 根据模型类型生成不同内容
        if model_type == 'growth_accounting':
            content = self._render_growth_accounting_report(report_data)
        elif model_type == 'okun_law':
            content = self._render_okun_law_report(report_data)
        elif model_type == 'phillips_curve':
            content = self._render_phillips_curve_report(report_data)
        else:
            content = f"# {model_type} 分析报告\n\n暂无详细分析内容"
        
        file_path = self._save_markdown(content, f"{model_type}_report")
        
        report_data['content'] = content
        report_data['file_path'] = str(file_path)
        
        return report_data
    
    def _generate_charts(self, ts: MacroTimeSeries, growth_df: pd.DataFrame) -> Dict[str, str]:
        """生成分析图表"""
        chart_dir = self.report_dir / 'charts'
        chart_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d')
        code = ts.meta.indicator.code
        
        paths = {}
        
        # 1. 时间序列图
        ts_path = chart_dir / f"{code}_timeseries_{timestamp}.png"
        self.visualizer.plot_time_series(ts, save_path=ts_path)
        paths['time_series'] = str(ts_path)
        
        # 2. 增长率图
        growth_path = chart_dir / f"{code}_growth_{timestamp}.png"
        self.visualizer.plot_growth_rate(ts, save_path=growth_path)
        paths['growth_rate'] = str(growth_path)
        
        return paths
    
    def _generate_forecast(self, ts: MacroTimeSeries, periods: int) -> Dict[str, Any]:
        """生成预测分析"""
        from core.predictor import Predictor
        
        try:
            predictor = Predictor()
            forecast_result = predictor.forecast(ts, periods=periods)
            
            return {
                'method': forecast_result.get('method', 'Unknown'),
                'forecast_values': forecast_result.get('forecast', []),
                'confidence_interval': forecast_result.get('confidence_interval', {}),
                'r_squared': forecast_result.get('r_squared', None)
            }
        except Exception as e:
            self.logger.error(f"预测生成失败: {e}")
            return {'error': str(e)}
    
    def _render_markdown_report(self, data: Dict[str, Any]) -> str:
        """渲染Markdown格式报告"""
        md = f"""# {data['indicator_name']} 深度分析报告

**报告生成时间**: {data['generated_time']}  
**数据时间范围**: {data['period']}  
**数据点数量**: {data['data_points']}

---

## 一、描述性统计

| 统计指标 | 数值 |
|---------|------|
"""
        
        # 添加描述性统计
        stats = data.get('descriptive_stats', {})
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                md += f"| {key} | {value:.4f} |\n"
        
        # 添加增长率分析
        growth = data.get('growth_analysis', {})
        if growth:
            md += f"""
---

## 二、增长率分析

- **最新同比增长率**: {growth.get('latest_yoy', 'N/A'):.2f}%
- **平均同比增长率**: {growth.get('avg_yoy', 'N/A'):.2f}%
- **增长趋势**: {growth.get('yoy_trend', 'N/A')}

"""
        
        # 添加平稳性检验
        stationarity = data.get('stationarity', {})
        if 'error' not in stationarity:
            md += f"""---

## 三、平稳性检验（ADF检验）

- **ADF统计量**: {stationarity.get('adf_statistic', 'N/A'):.4f}
- **P值**: {stationarity.get('p_value', 'N/A'):.4f}
- **检验结论**: {stationarity.get('interpretation', 'N/A')}

"""
        
        # 添加预测
        forecast = data.get('forecast', {})
        if 'error' not in forecast and forecast:
            md += f"""---

## 四、预测分析

- **预测方法**: {forecast.get('method', 'N/A')}
- **预测R²**: {forecast.get('r_squared', 'N/A'):.4f}

### 预测结果

"""
            forecast_values = forecast.get('forecast_values', [])
            for i, val in enumerate(forecast_values[:6]):  # 显示前6期
                md += f"- 第{i+1}期预测值: {val:.2f}\n"
        
        # 添加图表引用
        charts = data.get('charts', {})
        if charts:
            md += """
---

## 五、分析图表

"""
            for chart_name, chart_path in charts.items():
                md += f"### {chart_name.replace('_', ' ').title()}\n\n"
                md += f"![{chart_name}]({chart_path})\n\n"
        
        # 添加结论
        md += """
---

## 六、分析结论

"""
        md += self._generate_conclusion(data)
        
        return md
    
    def _generate_conclusion(self, data: Dict[str, Any]) -> str:
        """生成分析结论"""
        conclusion = []
        
        growth = data.get('growth_analysis', {})
        if growth:
            latest_yoy = growth.get('latest_yoy', 0)
            avg_yoy = growth.get('avg_yoy', 0)
            
            if latest_yoy > avg_yoy * 1.2:
                conclusion.append("当前增长率显著高于历史平均水平，显示强劲增长态势。")
            elif latest_yoy < avg_yoy * 0.8:
                conclusion.append("当前增长率低于历史平均水平，需关注增长动能减弱风险。")
            else:
                conclusion.append("当前增长率处于历史正常区间，经济运行总体平稳。")
        
        stationarity = data.get('stationarity', {})
        if stationarity.get('is_stationary', False):
            conclusion.append("该指标时间序列平稳，适合进行时间序列分析和预测。")
        else:
            conclusion.append("该指标存在非平稳特征，分析时需注意趋势性和季节性影响。")
        
        return "\n".join([f"- {c}" for c in conclusion])
    
    def _render_comparison_markdown(self, data: Dict[str, Any]) -> str:
        """渲染多指标对比报告"""
        md = f"""# 多指标对比分析报告

**报告生成时间**: {data['generated_time']}  
**指标数量**: {data['indicator_count']}

---

## 一、分析指标

"""
        for i, name in enumerate(data['indicators'], 1):
            md += f"{i}. {name}\n"
        
        md += """
---

## 二、相关性分析

### 相关性矩阵

"""
        
        # 添加相关性矩阵表格
        corr_matrix = data.get('correlation_matrix', {})
        if corr_matrix:
            # 表头
            codes = list(corr_matrix.keys())
            md += "| 指标 | " + " | ".join(codes) + " |\n"
            md += "|" + "---|" * (len(codes) + 1) + "\n"
            
            # 表格内容
            for code1 in codes:
                row = [code1]
                for code2 in codes:
                    val = corr_matrix.get(code1, {}).get(code2, 0)
                    row.append(f"{val:.3f}")
                md += "| " + " | ".join(row) + " |\n"
        
        # 添加图表
        if data.get('heatmap_path'):
            md += f"""

### 相关性热力图

![相关性热力图]({data['heatmap_path']})

"""
        
        if data.get('comparison_chart'):
            md += f"""

### 多指标对比图

![多指标对比图]({data['comparison_chart']})

"""
        
        return md
    
    def _render_growth_accounting_report(self, data: Dict[str, Any]) -> str:
        """渲染增长核算报告"""
        result = data['model_result']
        
        md = f"""# 经济增长核算分析报告

**报告生成时间**: {data['generated_time']}

---

## 一、模型概述

本报告基于索洛增长模型，将经济增长分解为资本、劳动和全要素生产率（TFP）的贡献。

模型公式：**g_Y = g_A + α·g_K + (1-α)·g_L**

其中：
- g_Y: GDP增长率
- g_K: 资本增长率
- g_L: 劳动增长率
- g_A: TFP增长率（索洛残差）
- α: 资本产出弹性

---

## 二、参数设置

- **资本产出弹性 (α)**: {result.get('capital_share', 0.4)}
- **劳动产出弹性 (1-α)**: {1 - result.get('capital_share', 0.4)}
- **分析时期**: {result.get('period', 'N/A')}

---

## 三、核算结果

| 指标 | 数值 |
|-----|------|
| 平均GDP增长率 | {result.get('avg_gdp_growth', 0):.2f}% |
| 资本贡献 | {result.get('avg_capital_contribution', 0):.2f}% |
| 劳动贡献 | {result.get('avg_labor_contribution', 0):.2f}% |
| TFP贡献 | {result.get('avg_tfp_growth', 0):.2f}% |

---

## 四、结论与启示

"""
        # 添加结论分析
        tfp_share = result.get('avg_tfp_growth', 0) / result.get('avg_gdp_growth', 1) * 100
        
        if tfp_share > 40:
            md += "- 经济增长质量较高，TFP贡献显著，技术进步和效率改善是重要驱动力。\n"
        elif tfp_share < 20:
            md += "- 经济增长主要依赖要素投入，需关注增长可持续性和效率提升。\n"
        else:
            md += "- 经济增长呈现要素投入与效率提升并重的格局。\n"
        
        return md
    
    def _render_okun_law_report(self, data: Dict[str, Any]) -> str:
        """渲染奥肯定律报告"""
        result = data['model_result']
        
        md = f"""# 奥肯定律分析报告

**报告生成时间**: {data['generated_time']}

---

## 一、模型概述

奥肯定律描述了GDP增长与失业率变动之间的经验关系：

**Δu = a + b·g_Y**

其中：
- Δu: 失业率变化
- g_Y: GDP增长率
- b: 奥肯系数（通常为负值）

---

## 二、估计结果

| 统计量 | 数值 |
|-------|------|
| 奥肯系数 | {result.get('okun_coefficient', 0):.4f} |
| 截距项 | {result.get('intercept', 0):.4f} |
| R² | {result.get('r_squared', 0):.4f} |
| t统计量 | {result.get('t_statistic', 0):.4f} |
| P值 | {result.get('p_value', 0):.4f} |
| 样本量 | {result.get('sample_size', 0)} |

---

## 三、回归方程

**{result.get('equation', 'N/A')}**

---

## 四、结论解释

{result.get('interpretation', 'N/A')}

---

## 五、注意事项

1. 中国失业率统计口径与西方国家存在差异
2. 劳动力市场存在结构性特征
3. 结果需结合具体国情谨慎解读
"""
        
        return md
    
    def _render_phillips_curve_report(self, data: Dict[str, Any]) -> str:
        """渲染菲利普斯曲线报告"""
        result = data['model_result']
        
        md = f"""# 菲利普斯曲线分析报告

**报告生成时间**: {data['generated_time']}

---

## 一、模型概述

本报告分析了通货膨胀率与失业率之间的关系。

模型类型：**{result.get('model_type', 'N/A')}**

---

## 二、估计结果

| 统计量 | 数值 |
|-------|------|
| 斜率系数 | {result.get('slope', 0):.4f} |
| 截距项 | {result.get('intercept', 0):.4f} |
| R² | {result.get('r_squared', 0):.4f} |
"""
        
        if result.get('model_type') == 'expectations_augmented':
            md += f"| 自然失业率 | {result.get('natural_unemployment', 0):.2f}% |\n"
        
        md += f"""
---

## 三、回归方程

**{result.get('equation', 'N/A')}**

---

## 四、结论解释

{result.get('interpretation', 'N/A')}
"""
        
        return md
    
    def _save_markdown(self, content: str, filename: str) -> Path:
        """保存Markdown文件"""
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = self.report_dir / f"{filename}_{timestamp}.md"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.logger.info(f"Markdown报告已保存: {file_path}")
        return file_path
    
    def _save_pdf(self, content: str, filename: str, chart_paths: Dict[str, str]) -> Path:
        """保存PDF文件（简化版，需要安装wkhtmltopdf或使用其他库）"""
        # 转换为HTML
        html_content = self._markdown_to_html(content)
        
        # 保存HTML
        html_path = self._save_html(html_content, filename)
        
        self.logger.warning("PDF生成需要额外依赖（如pdfkit/weasyprint），当前保存为HTML格式")
        return html_path
    
    def _save_html(self, content: str, filename: str) -> Path:
        """保存HTML文件"""
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = self.report_dir / f"{filename}_{timestamp}.html"
        
        html_content = self._markdown_to_html(content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML报告已保存: {file_path}")
        return file_path
    
    def _markdown_to_html(self, md_content: str) -> str:
        """Markdown转HTML（简化版）"""
        # 使用markdown库（如果可用）
        try:
            import markdown
            html_body = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
        except ImportError:
            # 简单替换
            html_body = md_content.replace('\n', '<br>\n')
        
        html_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>宏观经济分析报告</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', 'SimHei', Arial, sans-serif;
            line-height: 1.6;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        {html_body}
        <div class="footer">
            <p>本报告由国家宏观经济数据分析平台自动生成</p>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html_template
