"""
高性能大数据处理器
支持分布式计算、并行处理和大规模数据分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import logging
import time
from datetime import datetime

# 尝试导入Dask（如果可用）
try:
    import dask.dataframe as dd
    import dask.array as da
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

# 尝试导入Ray（如果可用）
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from config import Config
from models.time_series import MacroTimeSeries


class BigDataProcessor:
    """高性能大数据处理器"""
    
    def __init__(self, use_dask: bool = True, use_ray: bool = True, 
                 max_workers: Optional[int] = None):
        """
        初始化大数据处理器
        
        Args:
            use_dask: 是否使用Dask进行分布式计算
            use_ray: 是否使用Ray进行并行处理
            max_workers: 最大工作进程数
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.use_dask = use_dask and DASK_AVAILABLE
        self.use_ray = use_ray and RAY_AVAILABLE
        self.max_workers = max_workers or min(32, mp.cpu_count())
        
        # 初始化Ray（如果可用）
        if self.use_ray and not ray.is_initialized():
            try:
                ray.init(ignore_reinit_error=True)
                self.logger.info("Ray集群初始化成功")
            except Exception as e:
                self.logger.warning(f"Ray初始化失败: {e}")
                self.use_ray = False
    
    def process_large_dataset(self, data_dict: Dict[str, MacroTimeSeries], 
                            operations: List[str]) -> Dict[str, Any]:
        """
        处理大规模数据集
        
        Args:
            data_dict: 指标代码到时间序列的映射
            operations: 要执行的操作列表
            
        Returns:
            处理结果字典
        """
        start_time = time.time()
        self.logger.info(f"开始处理 {len(data_dict)} 个指标的大数据集")
        
        results = {}
        
        # 并行执行操作
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for operation in operations:
                if operation == 'correlation':
                    future = executor.submit(self._parallel_correlation, data_dict)
                    futures['correlation'] = future
                elif operation == 'clustering':
                    future = executor.submit(self._parallel_clustering, data_dict)
                    futures['clustering'] = future
                elif operation == 'forecasting':
                    future = executor.submit(self._parallel_forecasting, data_dict)
                    futures['forecasting'] = future
                elif operation == 'anomaly_detection':
                    future = executor.submit(self._parallel_anomaly_detection, data_dict)
                    futures['anomaly_detection'] = future
            
            # 收集结果
            for op_name, future in futures.items():
                try:
                    results[op_name] = future.result(timeout=300)  # 5分钟超时
                except Exception as e:
                    self.logger.error(f"{op_name}操作失败: {e}")
                    results[op_name] = {'error': str(e)}
        
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        results['processed_indicators'] = list(data_dict.keys())
        
        self.logger.info(f"大数据处理完成，耗时: {processing_time:.2f}秒")
        return results
    
    def _parallel_correlation(self, data_dict: Dict[str, MacroTimeSeries]) -> Dict[str, Any]:
        """并行计算相关性矩阵"""
        try:
            # 对齐所有时间序列
            aligned_data = self._align_time_series(data_dict)
            
            if self.use_dask:
                # 使用Dask计算相关性
                ddf = dd.from_pandas(aligned_data, npartitions=self.max_workers)
                correlation_matrix = ddf.corr().compute()
            else:
                # 使用Pandas计算相关性
                correlation_matrix = aligned_data.corr()
            
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'strongest_pairs': self._find_strongest_correlations(correlation_matrix),
                'success': True
            }
        except Exception as e:
            self.logger.error(f"相关性计算失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _parallel_clustering(self, data_dict: Dict[str, MacroTimeSeries]) -> Dict[str, Any]:
        """并行执行聚类分析"""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # 准备聚类数据
            cluster_data = []
            indicator_names = []
            
            for code, ts in data_dict.items():
                if len(ts) >= 12:  # 至少需要12个数据点
                    # 提取特征：均值、标准差、趋势、季节性等
                    features = self._extract_time_series_features(ts)
                    cluster_data.append(features)
                    indicator_names.append(code)
            
            if len(cluster_data) < 2:
                return {'success': False, 'error': '数据不足进行聚类'}
            
            # 标准化数据
            scaler = StandardScaler()
            cluster_data_scaled = scaler.fit_transform(cluster_data)
            
            # 执行K-means聚类
            n_clusters = min(5, len(cluster_data) // 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(cluster_data_scaled)
            
            # 组织聚类结果
            clusters = {}
            for i, (code, label) in enumerate(zip(indicator_names, cluster_labels)):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(code)
            
            return {
                'clusters': clusters,
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'inertia': kmeans.inertia_,
                'n_clusters': n_clusters,
                'success': True
            }
        except Exception as e:
            self.logger.error(f"聚类分析失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _parallel_forecasting(self, data_dict: Dict[str, MacroTimeSeries]) -> Dict[str, Any]:
        """并行执行预测"""
        try:
            from core.predictor import Predictor
            
            forecasts = {}
            
            # 使用进程池并行预测
            with ProcessPoolExecutor(max_workers=min(4, self.max_workers)) as executor:
                # 为每个指标创建预测任务
                tasks = []
                for code, ts in data_dict.items():
                    if len(ts) >= 20:  # 至少需要20个数据点
                        task = executor.submit(self._single_forecast, code, ts)
                        tasks.append((code, task))
                
                # 收集预测结果
                for code, task in tasks:
                    try:
                        result = task.result(timeout=120)  # 2分钟超时
                        forecasts[code] = result
                    except Exception as e:
                        self.logger.warning(f"{code}预测失败: {e}")
                        forecasts[code] = {'success': False, 'error': str(e)}
            
            return {
                'forecasts': forecasts,
                'successful_predictions': sum(1 for r in forecasts.values() if r.get('success', False)),
                'total_indicators': len(data_dict),
                'success': True
            }
        except Exception as e:
            self.logger.error(f"并行预测失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _parallel_anomaly_detection(self, data_dict: Dict[str, MacroTimeSeries]) -> Dict[str, Any]:
        """并行执行异常检测"""
        try:
            anomalies = {}
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for code, ts in data_dict.items():
                    future = executor.submit(self._detect_anomalies_single, ts)
                    futures[code] = future
                
                for code, future in futures.items():
                    try:
                        anomalies[code] = future.result(timeout=60)
                    except Exception as e:
                        self.logger.warning(f"{code}异常检测失败: {e}")
                        anomalies[code] = {'success': False, 'error': str(e)}
            
            return {
                'anomalies': anomalies,
                'total_detected': sum(1 for a in anomalies.values() 
                                    if a.get('success', False) and len(a.get('anomalous_points', [])) > 0),
                'success': True
            }
        except Exception as e:
            self.logger.error(f"异常检测失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _single_forecast(self, code: str, ts: MacroTimeSeries) -> Dict[str, Any]:
        """单个指标预测（用于进程池）"""
        try:
            from core.predictor import Predictor
            predictor = Predictor(method='auto')
            result = predictor.forecast(ts, periods=12)
            result['indicator_code'] = code
            result['success'] = True
            return result
        except Exception as e:
            return {'indicator_code': code, 'success': False, 'error': str(e)}
    
    def _detect_anomalies_single(self, ts: MacroTimeSeries) -> Dict[str, Any]:
        """单个时间序列的异常检测"""
        try:
            from scipy import stats
            
            values = ts.data['value'].values
            dates = ts.data['date'].values
            
            # 使用Z-score方法检测异常
            z_scores = np.abs(stats.zscore(values))
            threshold = 2.5  # 2.5个标准差阈值
            anomalous_indices = np.where(z_scores > threshold)[0]
            
            anomalous_points = []
            for idx in anomalous_indices:
                anomalous_points.append({
                    'date': str(dates[idx]),
                    'value': float(values[idx]),
                    'z_score': float(z_scores[idx])
                })
            
            return {
                'anomalous_points': anomalous_points,
                'anomaly_count': len(anomalous_points),
                'threshold': threshold,
                'success': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _align_time_series(self, data_dict: Dict[str, MacroTimeSeries]) -> pd.DataFrame:
        """对齐多个时间序列"""
        # 找到共同的时间范围
        all_dates = set()
        for ts in data_dict.values():
            all_dates.update(ts.data['date'])
        
        sorted_dates = sorted(list(all_dates))
        
        # 创建对齐的数据框
        aligned_data = pd.DataFrame({'date': sorted_dates})
        aligned_data = aligned_data.set_index('date')
        
        # 填充各指标数据
        for code, ts in data_dict.items():
            ts_df = ts.data.set_index('date')[['value']].rename(columns={'value': code})
            aligned_data = aligned_data.join(ts_df, how='left')
        
        # 前向填充缺失值
        aligned_data = aligned_data.fillna(method='ffill').fillna(method='bfill')
        
        return aligned_data
    
    def _extract_time_series_features(self, ts: MacroTimeSeries) -> List[float]:
        """提取时间序列特征用于聚类"""
        values = ts.data['value'].values
        
        features = [
            np.mean(values),           # 均值
            np.std(values),            # 标准差
            np.max(values),            # 最大值
            np.min(values),            # 最小值
            np.mean(np.diff(values)),  # 平均变化率
            np.std(np.diff(values)),   # 变化率标准差
        ]
        
        # 添加滞后特征
        for lag in [1, 3, 6]:
            if len(values) > lag:
                lag_corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                features.append(lag_corr if not np.isnan(lag_corr) else 0)
            else:
                features.append(0)
        
        return features
    
    def _find_strongest_correlations(self, corr_matrix: pd.DataFrame, 
                                   top_n: int = 10) -> List[Dict[str, Any]]:
        """找出最强的相关性对"""
        # 将相关性矩阵转换为列表格式
        correlations = []
        columns = corr_matrix.columns
        
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                corr_value = corr_matrix.iloc[i, j]
                if not np.isnan(corr_value):
                    correlations.append({
                        'indicator1': columns[i],
                        'indicator2': columns[j],
                        'correlation': abs(corr_value),
                        'direction': 'positive' if corr_value > 0 else 'negative'
                    })
        
        # 按相关性强度排序
        correlations.sort(key=lambda x: x['correlation'], reverse=True)
        return correlations[:top_n]
    
    def batch_process_indicators(self, indicator_codes: List[str], 
                               start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        批量处理大量指标
        
        Args:
            indicator_codes: 指标代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            处理结果
        """
        from core import DataManager
        
        dm = DataManager()
        successful_data = {}
        failed_data = []
        
        self.logger.info(f"开始批量处理 {len(indicator_codes)} 个指标")
        
        # 分批处理以控制内存使用
        batch_size = 50
        for i in range(0, len(indicator_codes), batch_size):
            batch = indicator_codes[i:i + batch_size]
            self.logger.info(f"处理批次 {i//batch_size + 1}/{(len(indicator_codes)-1)//batch_size + 1}")
            
            # 并行获取数据
            with ThreadPoolExecutor(max_workers=min(10, self.max_workers)) as executor:
                future_to_code = {
                    executor.submit(dm.fetch, code, start_date, end_date): code 
                    for code in batch
                }
                
                for future in future_to_code:
                    code = future_to_code[future]
                    try:
                        ts = future.result(timeout=30)
                        if ts is not None and len(ts) > 0:
                            successful_data[code] = ts
                        else:
                            failed_data.append(code)
                    except Exception as e:
                        self.logger.warning(f"获取 {code} 数据失败: {e}")
                        failed_data.append(code)
        
        self.logger.info(f"批量处理完成: 成功 {len(successful_data)}, 失败 {len(failed_data)}")
        
        return {
            'successful_data': successful_data,
            'failed_indicators': failed_data,
            'success_count': len(successful_data),
            'failure_count': len(failed_data)
        }


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 创建处理器实例
    processor = BigDataProcessor(use_dask=True, use_ray=True)
    
    # 示例：批量处理指标
    sample_indicators = ['gdp', 'cpi', 'ppi', 'pmi_manufacturing', 'retail_sales_yoy']
    
    # 批量获取数据
    batch_result = processor.batch_process_indicators(sample_indicators)
    print(f"成功获取 {batch_result['success_count']} 个指标的数据")
    
    if batch_result['successful_data']:
        # 大规模数据分析
        analysis_result = processor.process_large_dataset(
            batch_result['successful_data'],
            operations=['correlation', 'clustering', 'forecasting']
        )
        
        print("分析完成:")
        print(f"  处理时间: {analysis_result['processing_time']:.2f}秒")
        if 'correlation' in analysis_result and analysis_result['correlation']['success']:
            print(f"  发现强相关关系: {len(analysis_result['correlation']['strongest_pairs'])} 对")