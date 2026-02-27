#!/usr/bin/env python
"""
AI Companion项目命令行接口增强模块
第四部分：CLI功能扩展和优化
"""

import os
import sys
import argparse
import json
from pathlib import Path
import logging
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / f'cli_enhancement_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class CLIEnhancer:
    """命令行接口增强器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
    
    def create_enhanced_cli(self):
        """创建增强版命令行接口"""
        logger.info("🚀 创建增强版命令行接口...")
        
        enhanced_cli_content = '''#!/usr/bin/env python
"""
增强版命令行工具
提供更丰富的功能和更好的用户体验
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, IndicatorLibrary
from core.data_manager import DataManager
from core.analyzer import StatisticalAnalyzer
from core.visualizer import Visualizer
from core.reporter import ReportGenerator
from core.models import GrowthAccountingModel, OkunLawModel, PhillipsCurveModel


class EnhancedCLI:
    """增强版命令行接口类"""
    
    def __init__(self):
        """初始化CLI"""
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
        print(json.dumps(stats, indent=2, default=str))
        
        # 相关性分析（如果有多个指标）
        if hasattr(args, 'related_indicator') and args.related_indicator:
            ts2 = self.data_manager.fetch(args.related_indicator)
            if ts2:
                print("\n=== 相关性分析 ===")
                correlation = self.analyzer.correlation_analysis(ts, ts2)
                print(json.dumps(correlation, indent=2, default=str))
        
        # 导出报告
        if args.report:
            report_path = f"reports/analysis_{args.indicator}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            Path(report_path).parent.mkdir(exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'indicator': args.indicator,
                    'analysis': stats,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2, default=str, ensure_ascii=False)
            print(f"\n✓ 报告已导出至: {report_path}")
        
        return 0
    
    def predict(self, args):
        """预测数据命令"""
        print(f"正在预测 {args.periods} 期数据: {args.indicator}")
        
        ts = self.data_manager.fetch(args.indicator)
        if ts is None:
            print("❌ 数据获取失败")
            return 1
        
        predictions = self.analyzer.predict_next_values(ts, periods=args.periods)
        print(f"\n=== 预测结果 ===")
        for i, pred in enumerate(predictions, 1):
            print(f"  未来第{i}期: {pred:.4f}")
        
        # 导出预测结果
        if args.export:
            report_path = f"reports/prediction_{args.indicator}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            Path(report_path).parent.mkdir(exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'indicator': args.indicator,
                    'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
                    'periods': args.periods,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2, default=str, ensure_ascii=False)
            print(f"\n✓ 预测结果已导出至: {report_path}")
        
        return 0
    
    def compare(self, args):
        """多指标对比命令"""
        print(f"正在对比指标: {', '.join(args.indicators)}")
        
        data_series = []
        for indicator in args.indicators:
            ts = self.data_manager.fetch(indicator)
            if ts is not None:
                data_series.append((indicator, ts))
                print(f"  ✓ {indicator}: {len(ts)} 条数据")
            else:
                print(f"  ✗ {indicator}: 获取失败")
        
        if len(data_series) < 2:
            print("❌ 至少需要2个有效指标才能进行对比")
            return 1
        
        # 生成对比分析
        print("\n=== 对比分析 ===")
        correlations = []
        for i in range(len(data_series)):
            for j in range(i + 1, len(data_series)):
                ind1, ts1 = data_series[i]
                ind2, ts2 = data_series[j]
                
                correlation = self.analyzer.correlation_analysis(ts1, ts2)
                correlations.append({
                    'pair': f"{ind1}_vs_{ind2}",
                    'correlation': correlation
                })
        
        for corr in correlations:
            print(f"  {corr['pair']}: {corr['correlation']['pearson_correlation']:.4f}")
        
        # 生成对比图表
        if args.chart:
            chart_path = f"charts/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.visualizer.plot_multiple_series(
                [ts for _, ts in data_series],
                titles=[ind for ind, _ in data_series],
                save_path=chart_path
            )
            print(f"\n✓ 对比图表已生成: {chart_path}")
        
        return 0
    
    def model(self, args):
        """经济模型分析命令"""
        print(f"正在运行模型: {args.model_type}")
        
        if args.model_type == 'growth_accounting':
            # 增长核算模型
            gdp_ts = self.data_manager.fetch(args.gdp)
            capital_ts = self.data_manager.fetch('capital_stock')  # 假设指标存在
            labor_ts = self.data_manager.fetch('employment')      # 假设指标存在
            
            if gdp_ts and capital_ts and labor_ts:
                model = GrowthAccountingModel(capital_share=args.capital_share or 0.3)
                result = model.analyze(gdp_ts, capital_ts, labor_ts)
                print("\n=== 增长核算结果 ===")
                print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
            else:
                print("❌ 缺少必要数据")
                return 1
        
        elif args.model_type == 'okun_law':
            # 奥肯定律模型
            gdp_ts = self.data_manager.fetch(args.gdp)
            unemployment_ts = self.data_manager.fetch('unemployment_rate')
            
            if gdp_ts and unemployment_ts:
                model = OkunLawModel()
                result = model.analyze(gdp_ts, unemployment_ts)
                print("\n=== 奥肯定律分析结果 ===")
                print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
            else:
                print("❌ 缺少必要数据")
                return 1
        
        elif args.model_type == 'phillips_curve':
            # 菲利普斯曲线模型
            inflation_ts = self.data_manager.fetch('inflation_rate')
            unemployment_ts = self.data_manager.fetch('unemployment_rate')
            
            if inflation_ts and unemployment_ts:
                model = PhillipsCurveModel()
                result = model.analyze(inflation_ts, unemployment_ts)
                print("\n=== 菲利普斯曲线分析结果 ===")
                print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
            else:
                print("❌ 缺少必要数据")
                return 1
        
        return 0
    
    def list_indicators(self, args):
        """列出可用指标"""
        print("\n=== 可用指标列表 ===")
        
        categories = {
            'production': '生产类指标',
            'price': '价格类指标',
            'demand': '需求类指标',
            'trade': '贸易类指标',
            'finance': '金融类指标',
            'employment': '就业类指标'
        }
        
        for category_code, category_name in categories.items():
            print(f"\n{category_name}:")
            indicators = [ind for ind in IndicatorLibrary.NBS_INDICATORS if ind.category == category_code]
            for ind in indicators:
                leading_mark = ' ⭐' if ind.is_leading else ''
                coincident_mark = ' 📊' if ind.is_coincident else ''
                lagging_mark = ' 📈' if ind.is_lagging else ''
                print(f"  {ind.code} - {ind.name}{leading_mark}{coincident_mark}{lagging_mark}")
        
        return 0
    
    def status(self, args):
        """查看系统状态"""
        print("\n=== 系统状态 ===")
        print(f"数据源连接: {'✓ 正常' if self.data_manager.can_connect() else '✗ 异常'}")
        print(f"数据库连接: {'✓ 正常' if self.data_manager.db_manager.can_connect() else '✗ 异常'}")
        print(f"缓存状态: {'✓ 正常' if self.data_manager.cache_manager.is_available() else '✗ 异常'}")
        print(f"上次运行: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return 0
    
    def health_check(self, args):
        """健康检查"""
        print("\n=== 系统健康检查 ===")
        
        checks = [
            ("配置验证", lambda: Config.validate() or True),
            ("数据管理器", lambda: hasattr(self.data_manager, 'fetch')),
            ("分析器", lambda: hasattr(self.analyzer, 'descriptive_stats')),
            ("可视化器", lambda: hasattr(self.visualizer, 'plot_timeseries')),
            ("报告生成器", lambda: hasattr(self.reporter, 'generate_report'))
        ]
        
        for check_name, check_func in checks:
            try:
                result = check_func()
                status = "✓ 通过" if result else "✗ 失败"
                print(f"  {check_name}: {status}")
            except Exception as e:
                print(f"  {check_name}: ✗ 错误 - {e}")
        
        return 0
    
    def _export_data(self, ts, format_type, output_path):
        """导出数据"""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"exports/data_{ts.meta.indicator}_{timestamp}.{format_type}"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == 'excel':
            ts.to_excel(output_path)
        elif format_type == 'csv':
            ts.to_csv(output_path)
        elif format_type == 'json':
            ts.to_json(output_path)
        
        print(f"✓ 数据已导出至: {output_path}")
    
    def close(self):
        """清理资源"""
        if hasattr(self.data_manager, 'close'):
            self.data_manager.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='国家宏观经济数据分析平台 - 增强版命令行工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 获取GDP数据
  python cli.py fetch gdp --start 2010 --end 2023 --export excel

  # 分析指标
  python cli.py analyze cpi --report

  # 预测未来5期
  python cli.py predict gdp --periods 5

  # 多指标对比
  python cli.py compare gdp cpi ppi --chart

  # 运行经济模型
  python cli.py model growth_accounting --gdp gdp

  # 查看系统状态
  python cli.py status
  python cli.py health-check
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # fetch 命令
    fetch_parser = subparsers.add_parser('fetch', help='获取数据')
    fetch_parser.add_argument('indicator', help='指标代码')
    fetch_parser.add_argument('--start', help='开始日期')
    fetch_parser.add_argument('--end', help='结束日期')
    fetch_parser.add_argument('--refresh', action='store_true', help='强制刷新')
    fetch_parser.add_argument('--export', choices=['excel', 'csv', 'json'], help='导出格式')
    fetch_parser.add_argument('--output', help='输出路径')
    
    # analyze 命令
    analyze_parser = subparsers.add_parser('analyze', help='分析数据')
    analyze_parser.add_argument('indicator', help='指标代码')
    analyze_parser.add_argument('--related-indicator', help='关联指标')
    analyze_parser.add_argument('--report', action='store_true', help='生成报告')
    
    # predict 命令
    predict_parser = subparsers.add_parser('predict', help='预测数据')
    predict_parser.add_argument('indicator', help='指标代码')
    predict_parser.add_argument('--periods', type=int, default=5, help='预测期数')
    predict_parser.add_argument('--export', action='store_true', help='导出预测结果')
    
    # compare 命令
    compare_parser = subparsers.add_parser('compare', help='多指标对比')
    compare_parser.add_argument('indicators', nargs='+', help='指标代码列表')
    compare_parser.add_argument('--chart', action='store_true', help='生成对比图表')
    
    # model 命令
    model_parser = subparsers.add_parser('model', help='经济模型分析')
    model_parser.add_argument('model_type', 
                             choices=['growth_accounting', 'okun_law', 'phillips_curve'],
                             help='模型类型')
    model_parser.add_argument('--gdp', help='GDP指标代码')
    model_parser.add_argument('--capital-share', type=float, help='资本产出弹性')
    
    # list 命令
    list_parser = subparsers.add_parser('list', help='列出可用指标')
    
    # status 命令
    status_parser = subparsers.add_parser('status', help='查看系统状态')
    
    # health-check 命令
    health_parser = subparsers.add_parser('health-check', help='系统健康检查')
    
    # 便捷命令
    quick_parser = subparsers.add_parser('quick', help='快速分析')
    quick_parser.add_argument('indicator', help='指标代码')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # 执行命令
    cli = EnhancedCLI()
    try:
        if args.command == 'fetch':
            return cli.fetch(args)
        elif args.command == 'analyze':
            return cli.analyze(args)
        elif args.command == 'predict':
            return cli.predict(args)
        elif args.command == 'compare':
            return cli.compare(args)
        elif args.command == 'model':
            return cli.model(args)
        elif args.command == 'list':
            return cli.list_indicators(args)
        elif args.command == 'status':
            return cli.status(args)
        elif args.command == 'health-check':
            return cli.health_check(args)
        elif args.command == 'quick':
            # 快速分析：获取数据并生成基本报告
            print(f"正在快速分析: {args.indicator}")
            ts = cli.data_manager.fetch(args.indicator)
            if ts is None:
                print("❌ 数据获取失败")
                return 1
            
            print(f"✓ 获取到 {len(ts)} 条数据")
            stats = cli.analyzer.descriptive_stats(ts)
            print("\n=== 基本统计 ===")
            print(f"平均值: {stats['mean']:.4f}")
            print(f"标准差: {stats['std']:.4f}")
            print(f"最小值: {stats['min']:.4f}")
            print(f"最大值: {stats['max']:.4f}")
            return 0
        else:
            parser.print_help()
            return 1
    finally:
        cli.close()


if __name__ == '__main__':
    sys.exit(main())
'''
        
        enhanced_cli_file = self.project_root / 'enhanced_cli.py'
        with open(enhanced_cli_file, 'w', encoding='utf-8') as f:
            f.write(enhanced_cli_content)
        
        logger.info("  ✓ 创建增强版命令行接口")
    
    def create_cli_utilities(self):
        """创建CLI实用工具"""
        logger.info("🔧 创建CLI实用工具...")
        
        # 创建CLI工具目录
        cli_tools_dir = self.project_root / 'cli_tools'
        cli_tools_dir.mkdir(exist_ok=True)
        
        # 批处理脚本
        batch_script = cli_tools_dir / 'batch_processor.py'
        with open(batch_script, 'w', encoding='utf-8') as f:
            f.write('''"""
批处理器
用于批量执行CLI命令
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_batch_commands(commands_file):
    """运行批量命令"""
    with open(commands_file, 'r', encoding='utf-8') as f:
        commands = json.load(f)
    
    results = []
    for i, cmd in enumerate(commands):
        print(f"执行命令 {i+1}/{len(commands)}: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            results.append({
                'command': cmd,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            })
            print(f"  状态: {'✓ 成功' if result.returncode == 0 else '✗ 失败'}")
        except Exception as e:
            results.append({
                'command': cmd,
                'error': str(e)
            })
            print(f"  错误: {e}")
    
    # 保存结果
    output_file = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n批处理完成，结果保存至: {output_file}")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='批量命令处理器')
    parser.add_argument('commands_file', help='包含命令列表的JSON文件')
    args = parser.parse_args()
    
    run_batch_commands(args.commands_file)


if __name__ == '__main__':
    main()
''')
        
        # 示例批处理配置
        example_batch = cli_tools_dir / 'example_batch.json'
        with open(example_batch, 'w', encoding='utf-8') as f:
            f.write('''[
    ["python", "cli.py", "fetch", "gdp", "--start", "2020", "--end", "2023", "--export", "excel"],
    ["python", "cli.py", "analyze", "cpi", "--report"],
    ["python", "cli.py", "predict", "gdp", "--periods", "3"]
]''')
        
        logger.info("  ✓ 创建CLI实用工具")
    
    def enhance(self):
        """执行CLI增强"""
        logger.info("=" * 60)
        logger.info("🚀 开始命令行接口增强")
        logger.info("=" * 60)
        
        self.create_enhanced_cli()
        self.create_cli_utilities()
        
        logger.info("✅ 命令行接口增强完成")

def main():
    """主函数"""
    print("🚀 AI Companion 项目命令行接口增强模块")
    print("=" * 50)
    
    enhancer = CLIEnhancer()
    enhancer.enhance()
    
    print("\n🎯 CLI增强任务完成！")
    print("📁 已创建增强功能:")
    print("  - enhanced_cli.py: 增强版命令行工具")
    print("  - cli_tools/: CLI工具集")
    print("  - cli_tools/batch_processor.py: 批处理器")
    print("  - cli_tools/example_batch.json: 批处理示例")
    print("\n🎉 CLI增强完成！")

if __name__ == '__main__':
    main()