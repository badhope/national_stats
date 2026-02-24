"""
专业报告生成模块
生成PDF和Markdown分析报告
"""

from fpdf import FPDF
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict
import textwrap


class PDF(FPDF):
    """自定义PDF类"""
    
    def header(self):
        self.set_font('simhei', 'B', 12)
        self.cell(0, 10, 'National Statistics Analysis Report', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('simhei', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 注册中文字体 (需要将simhei.ttf放在项目根目录)
        self.font_path = Path(__file__).parent.parent / "simhei.ttf"
        
    def generate_markdown_report(self, data: Dict, analysis: Dict, predictions: Dict) -> Path:
        """生成Markdown报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"report_{timestamp}.md"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# 国家统计局数据分析报告\n\n")
            f.write(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("---\n\n")
            f.write("## 📊 数据概览\n\n")
            
            for key, df in data.items():
                f.write(f"### {key.upper()}\n\n")
                f.write(f"- 时间跨度: {df['year'].min()} - {df['year'].max()}\n")
                f.write(f"- 数据条目: {len(df)}\n\n")
            
            f.write("---\n\n")
            f.write("## 📈 预测分析\n\n")
            
            if 'gdp' in predictions:
                pred = predictions['gdp']
                f.write(f"### GDP预测\n\n")
                f.write(f"- 最佳模型: {pred['best_model']}\n")
                f.write(f"- R²得分: {pred['best_result']['metrics']['r2']:.4f}\n")
                f.write(f"- 未来5年预测值:\n")
                for x, y in zip(pred['best_result']['x'], pred['best_result']['y']):
                    f.write(f"  - {int(x)}: {y/10000:.2f} 万亿元\n")
                f.write("\n")
            
            f.write("---\n\n")
            f.write("*报告由 Python Data Analysis System 自动生成*\n")
        
        return filepath
    
    def generate_pdf_report(self, data: Dict, analysis: Dict, predictions: Dict, 
                           charts_dir: Path) -> Path:
        """生成PDF报告 (需要中文字体支持)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"report_{timestamp}.pdf"
        
        pdf = FPDF()
        pdf.add_page()
        
        # 尝试添加中文字体
        try:
            if self.font_path.exists():
                pdf.add_font('simhei', '', str(self.font_path), uni=True)
                pdf.set_font('simhei', '', 12)
            else:
                pdf.set_font('Arial', '', 12)
        except:
            pdf.set_font('Arial', '', 12)
        
        # 标题
        pdf.set_font_size(20)
        pdf.cell(0, 15, 'National Statistics Analysis Report', 0, 1, 'C')
        pdf.set_font_size(10)
        pdf.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'C')
        pdf.ln(10)
        
        # 内容
        pdf.set_font_size(14)
        pdf.cell(0, 10, '1. Data Overview', 0, 1)
        pdf.set_font_size(10)
        
        for key, df in data.items():
            pdf.cell(0, 8, f'- {key.upper()}: {len(df)} records', 0, 1)
        
        pdf.ln(5)
        
        # 插入图表
        if charts_dir.exists():
            pdf.set_font_size(14)
            pdf.cell(0, 10, '2. Visualization Charts', 0, 1)
            
            chart_files = list(charts_dir.glob("*.png"))
            for chart_file in chart_files[:3]:  # 只插入前3张
                try:
                    pdf.image(str(chart_file), x=10, w=180)
                    pdf.ln(5)
                except:
                    pass
        
        pdf.output(str(filepath))
        return filepath
