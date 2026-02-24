"""
命令行工具模式
支持无界面运行、自动化任务、批量导出
"""

import argparse
import sys
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from core.scraper import NationalBureauScraper
from core.predictor import ComprehensivePredictor
from core.reporter import ReportGenerator
from config import METRICS, EXPORT_DIR


def run_cli():
    parser = argparse.ArgumentParser(
        description='国家统计局数据分析系统 CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 获取GDP数据并导出CSV
  python cli.py fetch --metric gdp --start 2015 --end 2023 --export csv
  
  # 获取所有数据并生成报告
  python cli.py analyze --all --start 2010 --end 2024 --report
  
  # 预测未来5年GDP
  python cli.py predict --metric gdp --years 5
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # fetch 命令
    fetch_parser = subparsers.add_parser('fetch', help='获取数据')
    fetch_parser.add_argument('--metric', type=str, choices=list(METRICS.keys()) + ['all'], 
                             default='all', help='数据指标类型')
    fetch_parser.add_argument('--start', type=int, default=2010, help='起始年份')
    fetch_parser.add_argument('--end', type=int, default=2024, help='结束年份')
    fetch_parser.add_argument('--export', type=str, choices=['csv', 'json', 'excel'], 
                             help='导出格式')
    
    # analyze 命令
    analyze_parser = subparsers.add_parser('analyze', help='数据分析')
    analyze_parser.add_argument('--all', action='store_true', help='分析所有数据')
    analyze_parser.add_argument('--start', type=int, default=2010)
    analyze_parser.add_argument('--end', type=int, default=2024)
    analyze_parser.add_argument('--report', action='store_true', help='生成报告')
    
    # predict 命令
    predict_parser = subparsers.add_parser('predict', help='趋势预测')
    predict_parser.add_argument('--metric', type=str, required=True)
    predict_parser.add_argument('--years', type=int, default=5, help='预测未来年数')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    scraper = NationalBureauScraper()
    predictor = ComprehensivePredictor()
    reporter = ReportGenerator(EXPORT_DIR)
    
    if args.command == 'fetch':
        print(f"📊 正在获取数据: {args.metric} ({args.start}-{args.end})")
        
        if args.metric == 'all':
            data = scraper.fetch_all_parallel(args.start, args.end)
        else:
            df = scraper.fetch_data(args.metric, args.start, args.end)
            data = {args.metric: df}
        
        # 打印预览
        for key, df in data.items():
            print(f"\n{'='*50}")
            print(f" {METRICS[key]['name']}")
            print('='*50)
            print(df.head().to_string())
        
        # 导出
        if args.export:
            for key, df in data.items():
                if args.export == 'csv':
                    path = EXPORT_DIR / f"{key}_{args.start}_{args.end}.csv"
                    df.to_csv(path, index=False)
                elif args.export == 'json':
                    path = EXPORT_DIR / f"{key}_{args.start}_{args.end}.json"
                    df.to_json(path, orient='records', force_ascii=False)
                elif args.export == 'excel':
                    path = EXPORT_DIR / f"{key}_{args.start}_{args.end}.xlsx"
                    df.to_excel(path, index=False)
                print(f"\n✅ 已导出: {path}")
    
    elif args.command == 'analyze':
        print("📈 正在进行综合分析...")
        
        data = scraper.fetch_all_parallel(args.start, args.end)
        predictions = predictor.predict_all_metrics(data)
        
        # 控制台输出分析结果
        for key, pred in predictions.items():
            print(f"\n{key.upper()} 预测结果:")
            if 'best_model' in pred:
                print(f"  最佳模型: {pred['best_model']}")
                print(f"  R²得分: {pred['best_result']['metrics']['r2']:.4f}")
        
        if args.report:
            md_path = reporter.generate_markdown_report(data, {}, predictions)
            print(f"\n📄 报告已生成: {md_path}")
    
    elif args.command == 'predict':
        print(f"🔮 正在预测: {args.metric} 未来 {args.years} 年")
        
        # 获取历史数据
        df = scraper.fetch_data(args.metric, 2010, 2024)
        
        # 进行预测
        predictions = predictor.predict_all_metrics({args.metric: df}, args.years)
        
        if args.metric in predictions:
            pred = predictions[args.metric]
            print(f"\n预测结果:")
            for x, y in zip(pred['best_result']['x'], pred['best_result']['y']):
                print(f"  {int(x)}: {y:,.2f}")


if __name__ == "__main__":
    run_cli()
