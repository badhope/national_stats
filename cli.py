#!/usr/bin/env python
"""
命令行工具
提供完整的命令行交互接口
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import logging

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, IndicatorLibrary
from core import DataManager, StatisticalAnalyzer, Visualizer, ReportGenerator
from core.models import GrowthAccountingModel, OkunLawModel, PhillipsCurveModel
from core.predictor import Predictor
from core.fitter import AdvancedFitter, fit_trend_analysis


class CLI:
    """命令行接口类"""
    
    def __init__(self):
        """初始化CLI"""
        # 初始化日志
        logging.basicConfig(level=getattr(logging, Config.log.level))
        self.logger = logging.getLogger(__name__)
        
        self.data_manager = DataManager()
        self.analyzer = StatisticalAnalyzer()
        self.visualizer = Visualizer()
        self.reporter = ReportGenerator()
    
    def fetch(self, args):
        """获取数据命令"""
        print(f"正在获取数据: {args.indicator}")
        
        ts = self.data_manager.fetch(
            args.indicator,
            start_date=args.start,
            end_date=args.end,
            force_refresh=args.refresh
        )
        
        if ts is None:
            print("❌ 数据获取失败")
            return 1
        
        print(f"✓ 成功获取 {len(ts)} 条数据")
        print(f"  时间范围: {ts.meta.start_date} 至 {ts.meta.end_date}")
        
        # 显示基本统计信息
        if len(ts) > 0:
            latest_value = ts.data['value'].iloc[-1]
            print(f"  最新值: {latest_value:.2f} {ts.meta.unit}")
            
            if len(ts) > 1:
                growth_rate = ((latest_value / ts.data['value'].iloc[-2]) - 1) * 100
                print(f"  环比增长率: {growth_rate:.2f}%")
        
        # 导出
        if args.export:
            self._export_data(ts, args.export, args.output)
        
        return 0
    
    def analyze(self, args):
        """分析数据命令"""
        print(f"正在分析: {args.indicator}")
        
        ts = self.data_manager.fetch(args.indicator)
        if ts is None:
            print("❌ 数据获取失败")
            return 1
        
        # 描述性统计
        print("\n=== 描述性统计 ===")
        stats = self.analyzer.descriptive_stats(ts)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # 增长率分析
        print("\n=== 增长率分析 ===")
        try:
            growth_df = self.analyzer.calculate_growth_rates(ts)
            if len(growth_df) > 0:
                print(f"最新同比增长率: {growth_df['yoy'].iloc[-1]:.2f}%")
                print(f"平均同比增长率: {growth_df['yoy'].mean():.2f}%")
                print(f"增长率标准差: {growth_df['yoy'].std():.2f}%")
        except Exception as e:
            print(f"增长率分析失败: {e}")
        
        # 平稳性检验
        print("\n=== 平稳性检验 ===")
        try:
            adf_result = self.analyzer.adf_test(ts)
            print(f"ADF检验统计量: {adf_result.get('statistic', 'N/A'):.4f}")
            print(f"ADF检验结论: {adf_result.get('interpretation', 'N/A')}")
            print(f"P值: {adf_result.get('p_value', 'N/A'):.4f}")
        except Exception as e:
            print(f"平稳性检验失败: {e}")
        
        # 趋势分析
        if args.trend:
            print("\n=== 趋势分析 ===")
            try:
                x_vals = np.arange(len(ts.data))
                y_vals = ts.data['value'].values
                trend_result = fit_trend_analysis(x_vals, y_vals)
                print(f"最佳拟合方法: {trend_result['best_method']}")
                print(f"趋势方向: {trend_result['trend_direction']}")
                print(f"趋势强度: {trend_result['trend_strength']:.4f}")
                print(f"波动性: {trend_result['volatility']:.4f}")
            except Exception as e:
                print(f"趋势分析失败: {e}")
        
        # 生成报告
        if args.report:
            print("\n正在生成报告...")
            try:
                report_data = self.reporter.generate_indicator_report(ts)
                print(f"✓ 报告已生成: {report_data['file_path']}")
            except Exception as e:
                print(f"报告生成失败: {e}")
        
        return 0
    
    def predict(self, args):
        """预测命令"""
        print(f"正在预测: {args.indicator}")
        
        ts = self.data_manager.fetch(args.indicator)
        if ts is None:
            print("❌ 数据获取失败")
            return 1
        
        try:
            predictor = Predictor(method=args.method)
            
            # 执行预测
            result = predictor.forecast(ts, periods=args.periods)
            
            print(f"\n=== 预测结果 ({result.get('method', 'N/A')}) ===")
            print(f"预测期数: {result.get('periods', 0)}")
            
            if 'r_squared' in result.get('training_info', {}):
                print(f"训练R²: {result['training_info']['r_squared']:.4f}")
            
            # 显示评估指标
            if 'evaluation' in result:
                eval_metrics = result['evaluation']
                print(f"测试R²: {eval_metrics.get('r2', 0):.4f}")
                print(f"RMSE: {eval_metrics.get('rmse', 0):.4f}")
                print(f"MAE: {eval_metrics.get('mae', 0):.4f}")
            
            print("\n预测值:")
            forecast_dates = result.get('forecast_dates', [])
            forecast_values = result.get('forecast', [])
            
            for i, (date, val) in enumerate(zip(forecast_dates[:args.periods], forecast_values[:args.periods]), 1):
                print(f"  {date}: {val:.2f}")
            
            # 导出预测结果
            if args.export:
                self._export_prediction(result, args.output or f"prediction_{args.indicator}.json")
                
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return 1
        
        return 0
    
    def fit(self, args):
        """数据拟合命令"""
        print(f"正在进行数据拟合: {args.indicator}")
        
        ts = self.data_manager.fetch(args.indicator)
        if ts is None:
            print("❌ 数据获取失败")
            return 1
        
        try:
            # 准备数据
            x = np.arange(len(ts.data))
            y = ts.data['value'].values
            
            # 执行拟合
            fitter = AdvancedFitter()
            methods = args.methods.split(',') if args.methods else None
            result = fitter.fit(x, y, methods=methods)
            
            print(f"\n=== 拟合结果 ===")
            print(f"最佳方法: {result['best_method']}")
            print(f"R²得分: {result['r_squared']:.4f}")
            
            if 'parameters' in result:
                print("拟合参数:")
                for param, value in result['parameters'].items():
                    print(f"  {param}: {value:.6f}")
            
            # 趋势分析
            if 'trend_direction' in result:
                print(f"趋势方向: {result['trend_direction']}")
                print(f"趋势强度: {result['trend_strength']:.4f}")
                print(f"加速度: {result['acceleration']:.6f}")
            
            # 外推预测
            if args.extrapolate > 0:
                x_new = np.arange(len(x), len(x) + args.extrapolate)
                extrapolation = fitter.extrapolate(x_new)
                print(f"\n外推预测 ({args.extrapolate}期):")
                for i, (pred, lower, upper) in enumerate(zip(
                    extrapolation['predicted_values'],
                    extrapolation['confidence_lower'],
                    extrapolation['confidence_upper']
                ), 1):
                    print(f"  第{i}期: {pred:.2f} (置信区间: [{lower:.2f}, {upper:.2f}])")
            
            # 导出结果
            if args.export:
                self._export_fitting(result, args.output or f"fitting_{args.indicator}.json")
                
        except Exception as e:
            print(f"❌ 拟合失败: {e}")
            return 1
        
        return 0
    
    def compare(self, args):
        """多指标对比命令"""
        print(f"正在对比分析: {', '.join(args.indicators)}")
        
        ts_dict = self.data_manager.fetch_multiple(args.indicators)
        
        if not ts_dict:
            print("❌ 数据获取失败")
            return 1
        
        print(f"成功获取 {len(ts_dict)} 个指标的数据")
        
        # 相关性分析
        print("\n=== 相关性矩阵 ===")
        try:
            corr_matrix = self.analyzer.correlation_matrix(ts_dict)
            print(corr_matrix.round(3))
        except Exception as e:
            print(f"相关性分析失败: {e}")
        
        # 主成分分析
        if args.pca:
            print("\n=== 主成分分析 ===")
            try:
                pca_result = self.analyzer.pca_analysis(ts_dict)
                print(f"前两个主成分解释方差比: {pca_result['explained_variance_ratio'][:2]}")
            except Exception as e:
                print(f"PCA分析失败: {e}")
        
        # 生成报告
        if args.report:
            try:
                report_data = self.reporter.generate_comparison_report(ts_dict)
                print(f"\n✓ 报告已生成: {report_data['file_path']}")
            except Exception as e:
                print(f"报告生成失败: {e}")
        
        return 0
    
    def model(self, args):
        """经济模型命令"""
        print(f"正在运行模型: {args.model_type}")
        
        if args.model_type == 'growth_accounting':
            try:
                model = GrowthAccountingModel(capital_share=args.capital_share or 0.4)
                
                # 获取所需数据
                gdp_ts = self.data_manager.fetch(args.gdp or 'gdp')
                labor_ts = self.data_manager.fetch(args.labor or 'labor')
                capital_ts = self.data_manager.fetch(args.capital or 'capital')
                
                if not all([gdp_ts, labor_ts, capital_ts]):
                    print("❌ 缺少必要的数据")
                    return 1
                
                # 运行模型
                result = model.calculate(gdp_ts, labor_ts, capital_ts)
                
                print("\n=== 增长核算结果 ===")
                print(f"GDP增长率: {result['gdp_growth']:.2f}%")
                print(f"劳动贡献: {result['labor_contribution']:.2f}%")
                print(f"资本贡献: {result['capital_contribution']:.2f}%")
                print(f"TFP增长率: {result['tfp_growth']:.2f}%")
                print(f"规模效应: {result['scale_effect']:.2f}%")
                
            except Exception as e:
                print(f"❌ 增长核算失败: {e}")
                return 1
                
        elif args.model_type == 'okun_law':
            try:
                model = OkunLawModel()
                
                gdp_ts = self.data_manager.fetch(args.gdp or 'gdp_yoy')
                unemployment_ts = self.data_manager.fetch(args.unemployment or 'urban_unemployment_rate')
                
                if not all([gdp_ts, unemployment_ts]):
                    print("❌ 缺少必要的数据")
                    return 1
                
                result = model.estimate_relationship(gdp_ts, unemployment_ts)
                
                print("\n=== 奥肯定律结果 ===")
                print(f"奥肯系数: {result['okun_coefficient']:.4f}")
                print(f"R²: {result['r_squared']:.4f}")
                print(f"失业率自然水平: {result['natural_unemployment_rate']:.2f}%")
                
            except Exception as e:
                print(f"❌ 奥肯定律分析失败: {e}")
                return 1
                
        elif args.model_type == 'phillips_curve':
            try:
                model = PhillipsCurveModel()
                
                inflation_ts = self.data_manager.fetch(args.inflation or 'cpi')
                unemployment_ts = self.data_manager.fetch(args.unemployment or 'urban_unemployment_rate')
                
                if not all([inflation_ts, unemployment_ts]):
                    print("❌ 缺少必要的数据")
                    return 1
                
                result = model.estimate_phillips_curve(inflation_ts, unemployment_ts)
                
                print("\n=== 菲利普斯曲线结果 ===")
                print(f"通胀对失业的敏感性: {result['inflation_sensitivity']:.4f}")
                print(f"R²: {result['r_squared']:.4f}")
                print(f"非加速通胀失业率: {result['nairu']:.2f}%")
                
            except Exception as e:
                print(f"❌ 菲利普斯曲线分析失败: {e}")
                return 1
        
        return 0
    
    def list_indicators(self, args):
        """列出指标命令"""
        print("=== 可用指标列表 ===")
        
        if args.category:
            indicators = self.data_manager.list_indicators(category=args.category)
        else:
            indicators = self.data_manager.list_indicators()
        
        # 按类别分组显示
        categories = {}
        for ind in indicators:
            cat = ind['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(ind)
        
        for category, inds in categories.items():
            print(f"\n{category.upper()}:")
            for ind in inds:
                print(f"  {ind['code']:20} {ind['name']:30} ({ind['frequency']})")
        
        return 0
    
    def _export_data(self, ts, format_type: str, output_path: str = None):
        """导出数据"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data_export_{ts.meta.indicator_code}_{timestamp}.{format_type}"
        
        try:
            if format_type == 'csv':
                ts.to_csv(output_path)
            elif format_type == 'excel':
                ts.to_excel(output_path)
            elif format_type == 'json':
                ts.to_json(output_path)
            
            print(f"✓ 数据已导出到: {output_path}")
        except Exception as e:
            print(f"❌ 导出失败: {e}")
    
    def _export_prediction(self, result: dict, output_path: str):
        """导出预测结果"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✓ 预测结果已导出到: {output_path}")
        except Exception as e:
            print(f"❌ 导出预测结果失败: {e}")
    
    def _export_fitting(self, result: dict, output_path: str):
        """导出拟合结果"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✓ 拟合结果已导出到: {output_path}")
        except Exception as e:
            print(f"❌ 导出拟合结果失败: {e}")
    
    def run(self, args_list: Optional[List[str]] = None):
        """运行CLI"""
        parser = self._create_parser()
        args = parser.parse_args(args_list)
        
        # 执行对应命令
        if hasattr(self, args.command):
            return getattr(self, args.command)(args)
        else:
            parser.print_help()
            return 1
    
    def _create_parser(self):
        """创建命令行参数解析器"""
        parser = argparse.ArgumentParser(
            description="宏观经济数据分析工具",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
使用示例:
  python cli.py fetch gdp --start 2020-01 --end 2023-12
  python cli.py analyze cpi --trend --report
  python cli.py predict gdp --periods 12 --method auto
  python cli.py fit gdp --methods polynomial,exponential --extrapolate 6
  python cli.py compare gdp cpi pmi_manufacturing --pca --report
  python cli.py model growth_accounting --capital-share 0.4
  python cli.py list-indicators --category production
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='可用命令')
        
        # 获取数据命令
        fetch_parser = subparsers.add_parser('fetch', help='获取指标数据')
        fetch_parser.add_argument('indicator', help='指标代码')
        fetch_parser.add_argument('--start', help='开始日期 (YYYY-MM)')
        fetch_parser.add_argument('--end', help='结束日期 (YYYY-MM)')
        fetch_parser.add_argument('--refresh', action='store_true', help='强制刷新数据')
        fetch_parser.add_argument('--export', choices=['csv', 'excel', 'json'], help='导出格式')
        fetch_parser.add_argument('--output', help='输出文件路径')
        
        # 分析命令
        analyze_parser = subparsers.add_parser('analyze', help='分析指标数据')
        analyze_parser.add_argument('indicator', help='指标代码')
        analyze_parser.add_argument('--trend', action='store_true', help='执行趋势分析')
        analyze_parser.add_argument('--report', action='store_true', help='生成分析报告')
        
        # 预测命令
        predict_parser = subparsers.add_parser('predict', help='预测指标走势')
        predict_parser.add_argument('indicator', help='指标代码')
        predict_parser.add_argument('--periods', type=int, default=12, help='预测期数')
        predict_parser.add_argument('--method', choices=['auto', 'arima', 'prophet', 'xgboost', 'ensemble'], 
                                  default='auto', help='预测方法')
        predict_parser.add_argument('--export', action='store_true', help='导出预测结果')
        predict_parser.add_argument('--output', help='输出文件路径')
        
        # 拟合命令
        fit_parser = subparsers.add_parser('fit', help='数据拟合分析')
        fit_parser.add_argument('indicator', help='指标代码')
        fit_parser.add_argument('--methods', help='拟合方法列表 (逗号分隔)')
        fit_parser.add_argument('--extrapolate', type=int, default=0, help='外推期数')
        fit_parser.add_argument('--export', action='store_true', help='导出拟合结果')
        fit_parser.add_argument('--output', help='输出文件路径')
        
        # 对比命令
        compare_parser = subparsers.add_parser('compare', help='多指标对比分析')
        compare_parser.add_argument('indicators', nargs='+', help='指标代码列表')
        compare_parser.add_argument('--pca', action='store_true', help='执行主成分分析')
        compare_parser.add_argument('--report', action='store_true', help='生成对比报告')
        
        # 模型命令
        model_parser = subparsers.add_parser('model', help='经济模型分析')
        model_parser.add_argument('model_type', choices=['growth_accounting', 'okun_law', 'phillips_curve'],
                                help='模型类型')
        model_parser.add_argument('--gdp', help='GDP指标代码')
        model_parser.add_argument('--labor', help='劳动力指标代码')
        model_parser.add_argument('--capital', help='资本指标代码')
        model_parser.add_argument('--unemployment', help='失业率指标代码')
        model_parser.add_argument('--inflation', help='通胀率指标代码')
        model_parser.add_argument('--capital-share', type=float, help='资本份额参数')
        
        # 列出指标命令
        list_parser = subparsers.add_parser('list-indicators', help='列出可用指标')
        list_parser.add_argument('--category', help='筛选类别')
        
        return parser


# ==================== 主程序入口 ====================

def main():
    """主函数"""
    cli = CLI()
    exit_code = cli.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    # 导入numpy用于数据分析
    import numpy as np
    main()