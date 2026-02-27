"""
可视化模块
提供丰富的图表绘制功能，支持静态和交互式可视化
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 导入配置和模型
import sys
sys.path.append('..')
from config import Config, VisualizationConfig
from models.time_series import MacroTimeSeries


class Visualizer:
    """可视化工具类"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        初始化可视化工具
        
        Args:
            config: 可视化配置
        """
        self.config = config or Config.visualization
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 设置样式
        self._setup_style()
    
    def _setup_style(self):
        """设置图表样式"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = [self.config.chinese_font, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置 Seaborn 样式
        sns.set_palette(self.config.color_palette)
        sns.set_style("whitegrid")
    
    # ==================== 1. 基础图表 ====================
    
    def plot_time_series(self, ts: MacroTimeSeries,
                        title: Optional[str] = None,
                        show_ma: bool = True,
                        ma_windows: List[int] = [3, 12],
                        figsize: Optional[Tuple[int, int]] = None,
                        save_path: Optional[Union[str, Path]] = None,
                        show_plotly: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        绘制时间序列图
        
        Args:
            ts: 时间序列对象
            title: 图表标题
            show_ma: 是否显示移动平均线
            ma_windows: 移动平均窗口列表
            figsize: 图表尺寸
            save_path: 保存路径
            show_plotly: 是否使用 Plotly 绘制交互式图表
        
        Returns:
            图表对象
        """
        if show_plotly:
            return self._plot_time_series_plotly(ts, title, show_ma, ma_windows)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize or (self.config.figure_width, self.config.figure_height))
        
        # 绘制原始数据
        ax.plot(ts.data.index, ts.data['value'], label=ts.name, linewidth=2, color='#2E86AB')
        
        # 绘制移动平均
        if show_ma:
            colors = ['#A23B72', '#F18F01', '#C73E1D']
            for i, window in enumerate(ma_windows):
                ma = ts.data['value'].rolling(window=window).mean()
                ax.plot(ts.data.index, ma, label=f'MA{window}', 
                       linewidth=1.5, linestyle='--', color=colors[i % len(colors)])
        
        # 设置标签和标题
        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel(ts.meta.indicator.unit, fontsize=12)
        ax.set_title(title or f'{ts.name} 变化趋势', fontsize=14, fontweight='bold')
        
        # 格式化X轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        # 添加图例和网格
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def _plot_time_series_plotly(self, ts: MacroTimeSeries,
                                 title: Optional[str],
                                 show_ma: bool,
                                 ma_windows: List[int]) -> go.Figure:
        """Plotly 版本的时间序列图"""
        fig = go.Figure()
        
        # 原始数据
        fig.add_trace(go.Scatter(
            x=ts.data.index,
            y=ts.data['value'],
            mode='lines',
            name=ts.name,
            line=dict(color='#2E86AB', width=2)
        ))
        
        # 移动平均
        if show_ma:
            colors = ['#A23B72', '#F18F01', '#C73E1D']
            for i, window in enumerate(ma_windows):
                ma = ts.data['value'].rolling(window=window).mean()
                fig.add_trace(go.Scatter(
                    x=ts.data.index,
                    y=ma,
                    mode='lines',
                    name=f'MA{window}',
                    line=dict(color=colors[i % len(colors)], width=1.5, dash='dash')
                ))
        
        fig.update_layout(
            title=title or f'{ts.name} 变化趋势',
            xaxis_title='时间',
            yaxis_title=ts.meta.indicator.unit,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_growth_rate(self, ts: MacroTimeSeries,
                        figsize: Optional[Tuple[int, int]] = None,
                        save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        绘制增长率对比图（同比和环比）
        
        Args:
            ts: 时间序列对象
            figsize: 图表尺寸
            save_path: 保存路径
        
        Returns:
            图表对象
        """
        from core.analyzer import StatisticalAnalyzer
        
        # 计算增长率
        analyzer = StatisticalAnalyzer()
        df = analyzer.calculate_growth_rates(ts)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize or (12, 8), 
                                        gridspec_kw={'height_ratios': [2, 1]})
        
        # 上图：原始值
        ax1.plot(df.index, df['value'], label=ts.name, linewidth=2, color='#2E86AB')
        ax1.set_title(f'{ts.name} 变化趋势', fontsize=14, fontweight='bold')
        ax1.set_ylabel(ts.meta.indicator.unit, fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 下图：增长率
        ax2.bar(df.index, df['yoy'], label='同比增长率', color='#A23B72', alpha=0.7, width=20)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_title('同比增长率', fontsize=12)
        ax2.set_ylabel('增长率 (%)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    # ==================== 2. 多指标对比图 ====================
    
    def plot_multiple_series(self, ts_dict: Dict[str, MacroTimeSeries],
                           title: str = "多指标对比分析",
                           normalize: bool = False,
                           figsize: Optional[Tuple[int, int]] = None,
                           save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        绘制多指标对比图
        
        Args:
            ts_dict: 指标字典
            title: 图表标题
            normalize: 是否标准化（便于对比）
            figsize: 图表尺寸
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=figsize or (14, 7))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(ts_dict)))
        
        for (code, ts), color in zip(ts_dict.items(), colors):
            values = ts.values
            
            # 标准化
            if normalize:
                values = (values - values.mean()) / values.std()
            
            ax.plot(ts.data.index, values, label=ts.name, linewidth=2, color=color)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel('标准化值' if normalize else '数值', fontsize=12)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame,
                                title: str = "指标相关性热力图",
                                figsize: Optional[Tuple[int, int]] = None,
                                save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        绘制相关性热力图
        
        Args:
            corr_matrix: 相关性矩阵
            title: 图表标题
            figsize: 图表尺寸
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=figsize or (10, 8))
        
        # 绘制热力图
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='RdYlBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    # ==================== 3. 结构分析图 ====================
    
    def plot_structure_chart(self, data: pd.DataFrame,
                           chart_type: str = 'stacked_bar',
                           title: str = "结构分析图",
                           figsize: Optional[Tuple[int, int]] = None,
                           save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        绘制结构分析图（堆叠柱状图或堆叠面积图）
        
        Args:
            data: 数据框，行是时间，列是各组成部分
            chart_type: 图表类型
            title: 图表标题
            figsize: 图表尺寸
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=figsize or (12, 7))
        
        if chart_type == 'stacked_bar':
            data.plot(kind='bar', stacked=True, ax=ax, colormap='Set2', width=0.8)
        elif chart_type == 'stacked_area':
            data.plot(kind='area', stacked=True, ax=ax, alpha=0.7, colormap='Set2')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel('数值', fontsize=12)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    # ==================== 4. 回归分析图 ====================
    
    def plot_regression(self, y: MacroTimeSeries, x: MacroTimeSeries,
                       regression_result: Optional[Dict] = None,
                       figsize: Optional[Tuple[int, int]] = None,
                       save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        绘制回归分析图
        
        Args:
            y: 因变量
            x: 自变量
            regression_result: 回归结果（可选，如果不传则自动计算）
            figsize: 图表尺寸
            save_path: 保存路径
        """
        from core.analyzer import StatisticalAnalyzer
        
        # 合并数据
        df = pd.DataFrame({
            'x': x.values,
            'y': y.values
        }).dropna()
        
        # 如果没有回归结果，则计算
        if regression_result is None:
            analyzer = StatisticalAnalyzer()
            regression_result = analyzer.ols_regression(y, [x])
        
        fig, ax = plt.subplots(figsize=figsize or (10, 7))
        
        # 散点图
        ax.scatter(df['x'], df['y'], alpha=0.6, s=50, color='#2E86AB', label='实际值')
        
        # 回归线
        if 'error' not in regression_result:
            slope = regression_result['coefficients'].get(x.meta.indicator.code, 0)
            intercept = regression_result['coefficients'].get('const', 0)
            
            x_line = np.linspace(df['x'].min(), df['x'].max(), 100)
            y_line = slope * x_line + intercept
            
            ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'拟合线 (R²={regression_result["rsquared"]:.3f})')
            
            # 添加回归方程
            equation = f'y = {slope:.4f}x + {intercept:.4f}'
            ax.text(0.05, 0.95, equation, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel(x.name, fontsize=12)
        ax.set_ylabel(y.name, fontsize=12)
        ax.set_title(f'{y.name} 与 {x.name} 关系', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    # ==================== 5. 景气指数图 ====================
    
    def plot_composite_index(self, leading: pd.Series, coincident: pd.Series, lagging: pd.Series,
                           title: str = "经济景气指数",
                           figsize: Optional[Tuple[int, int]] = None,
                           save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        绘制景气指数三线图
        
        Args:
            leading: 先行指数
            coincident: 一致指数
            lagging: 滞后指数
            title: 图表标题
            figsize: 图表尺寸
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=figsize or (14, 7))
        
        # 绘制三条线
        ax.plot(leading.index, leading, label='先行指数', linewidth=2.5, color='#E74C3C')
        ax.plot(coincident.index, coincident, label='一致指数', linewidth=2.5, color='#3498DB')
        ax.plot(lagging.index, lagging, label='滞后指数', linewidth=2.5, color='#2ECC71')
        
        # 添加100基准线
        ax.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # 添加经济周期阴影区域（示例）
        # 实际应用中应根据数据判断周期
        ax.fill_between(coincident.index, 100, coincident, 
                       where=(coincident >= 100), alpha=0.2, color='green', label='扩张期')
        ax.fill_between(coincident.index, 100, coincident, 
                       where=(coincident < 100), alpha=0.2, color='red', label='收缩期')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel('指数', fontsize=12)
        ax.legend(loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    # ==================== 辅助方法 ====================
    
    def _save_figure(self, fig: plt.Figure, path: Union[str, Path]):
        """保存图表到文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(path, dpi=self.config.export_quality, bbox_inches='tight')
        self.logger.info(f"图表已保存: {path}")
    
    def close_all(self):
        """关闭所有图表"""
        plt.close('all')


# ==================== 使用示例 ====================

if __name__ == "__main__":
    from core.data_manager import DataManager
    from core.analyzer import StatisticalAnalyzer
    
    # 初始化
    viz = Visualizer()
    dm = DataManager()
    analyzer = StatisticalAnalyzer()
    
    # 获取数据
    gdp_ts = dm.fetch("gdp")
    
    if gdp_ts:
        # 1. 时间序列图
        fig1 = viz.plot_time_series(gdp_ts, show_ma=True)
        plt.show()
        
        # 2. 增长率图
        fig2 = viz.plot_growth_rate(gdp_ts)
        plt.show()
    
    dm.close()
    viz.close_all()
