#!/usr/bin/env python3
"""
Optuna超参数优化器模块 (完全由Config驱动的最终修正版)
"""
import os
import copy
import datetime
import gc
import signal
import sys
import time

import optuna
import optuna.visualization as vis
import pandas as pd
import psutil
import torch
import numpy as np
import random
from optuna.exceptions import TrialPruned

class OptunaHyperparameterOptimizer:
    """一个完全由外部配置驱动的Optuna超参数优化器"""

    def __init__(self, base_config, train_function, full_config_from_yaml):
        self.base_config = base_config
        self.train_function = train_function
        self.full_config = full_config_from_yaml
        self.optuna_config = self.full_config['optuna']
        self.output_dir = self.base_config.output_dir
        self.study = None
        self.storage_name = None
        self.db_path = None

    # [修正2 新增]：添加一个独立的set_seed方法，用于设置所有库的随机种子
    def _set_seed(self, seed):
        """为所有相关的库设置随机种子以保证可复现性。"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # 为了完全的可复现性，可以开启以下设置，但可能会牺牲一些性能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _setup_storage(self):
        """根据配置设置数据库存储."""
        os.makedirs(self.output_dir, exist_ok=True)
        self.db_path = os.path.join(self.output_dir, f"{self.optuna_config['study_name']}.db")
        self.storage_name = f"sqlite:///{self.db_path}"
        print(f"💾 存储路径 (Storage): {self.storage_name}")

    def _create_pruner(self):
        """动态创建剪枝器."""
        pruner_config = self.optuna_config.get('pruner', {})
        if not pruner_config.get('enable', False):
            print("✂️ 剪枝器 (Pruner): 已禁用")
            return None

        pruner_type = pruner_config.get('type', 'MedianPruner').lower()
        
        params = pruner_config.copy()
        params.pop('type', None)
        params.pop('enable', None)

        if pruner_type == 'medianpruner':
            pruner = optuna.pruners.MedianPruner(**params)
            print(f"✂️ 剪枝器 (Pruner): 已从配置文件加载 MedianPruner (参数: {params})")
            return pruner
        
        print(f"⚠️ 未知的剪枝器类型: {pruner_type}, 将不使用剪枝器。")
        return None

    def _create_sampler(self):
        """动态创建采样器."""
        sampler_config = self.optuna_config.get('sampler', {})
        sampler_type = sampler_config.get('type', 'TPESampler').lower()
        
        params = sampler_config.get('params', {})

        if sampler_type == 'tpesampler':
            sampler = optuna.samplers.TPESampler(**params)
            print(f"🔭 采样器 (Sampler): 已从配置文件加载 TPESampler (参数: {params})")
            return sampler
        
        print(f"⚠️ 未知的采样器类型: {sampler_type}, 将使用Optuna默认采样器。")
        return None

    def _create_study(self):
        """完全根据配置动态创建Optuna study."""
        self._setup_storage()
        pruner = self._create_pruner()
        sampler = self._create_sampler()
        
        self.study = optuna.create_study(
            study_name=self.optuna_config['study_name'],
            storage=self.storage_name,
            load_if_exists=self.optuna_config.get('resume', {}).get('enable', True),
            direction=self.optuna_config.get('direction', 'minimize'),
            pruner=pruner,
            sampler=sampler
        )
        print(f"📈 研究已加载，历史试验次数: {len(self.study.trials)}")

    def _objective(self, trial: optuna.trial.Trial) -> float:
        """优化目标函数，搜索空间和种子均由config和trial动态决定."""
        run_config = copy.deepcopy(self.base_config)
        
        # [修正2 实现]: 为每个Trial设置独立的、可复现的随机种子
        trial_seed = self.base_config.seed + trial.number*10
        run_config.seed = trial_seed
        self._set_seed(trial_seed)
        
        search_space = self.optuna_config['search_space']
        for param, settings in search_space.items():
            suggest_type = settings['type']
            if suggest_type == 'categorical':
                value = trial.suggest_categorical(param, settings['choices'])
            elif suggest_type == 'float':
                value = trial.suggest_float(param, settings['low'], settings['high'], log=settings.get('log', False))
            elif suggest_type == 'int':
                value = trial.suggest_int(param, settings['low'], settings['high'], step=settings.get('step', 1))
            else:
                raise ValueError(f"不支持的搜索空间类型: {suggest_type}")
                        # ========== 修改：改进参数设置逻辑 ==========
            if hasattr(run_config, param):
                setattr(run_config, param, value)
            elif param == 'weight_decay':
                run_config.optimizer_args['weight_decay'] = value
            elif param in ['coulomb_param', 'london_param', 'pauli_param', 'R_grid']:
                # PotNet参数：直接设置到config对象
                setattr(run_config, param, value)
            else:
                # 对于其他未识别的参数，也尝试设置（而不是仅仅警告）
                setattr(run_config, param, value)
                print(f"注意: 参数 '{param}' 已设置，值为 {value}")
            # ==========================================

        print(f"\n🚀 Starting Trial #{trial.number} (Seed: {trial_seed})")
        print("  Parameters:")
        for key, value in trial.params.items():
            print(f"    - {key}: {value}")

        try:
            best_val_mae = self.train_function(run_config, printnet=False, trial=trial)
            print(f"\n🏁 Trial #{trial.number} completed! Result: {best_val_mae:.6f}")
            return best_val_mae
        except TrialPruned:
            print(f"\n✂️ Trial #{trial.number} pruned!")
            raise
        except Exception as e:
            print(f"❌ Trial #{trial.number} failed with error: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')

    def run(self):
        """执行完整的超参数优化流程."""
        self._create_study()

        # [修正1]: 直接从配置中读取试验次数，不再使用带默认值的 .get()
        # 这样代码意图更清晰，完全遵循您的config文件。
        n_trials = self.optuna_config['n_trials']
        # [最终修正] 替换后的代码
        n_trials = self.optuna_config['n_trials']

        # 从配置中读取超时设置，并进行健壮性处理
        timeout_from_config = self.optuna_config.get('timeout')

        # 如果从YAML读入的是字符串'None'或空值，则将其转换成Python的None对象
        if timeout_from_config is None or str(timeout_from_config).lower() in ['none', 'null', '']:
            timeout = None
        else:
            # 否则，尝试将其转换为浮点数（代表秒数）
            try:
                timeout = float(timeout_from_config)
            except (ValueError, TypeError):
                print(f"⚠️ 超时设置 '{timeout_from_config}' 无法识别，将不设置超时。")
                timeout = None

        print(f"\n🚀 开始优化... 将运行 {n_trials} 次试验。")
        
        try:
            self.study.optimize(
                self._objective,
                n_trials=n_trials,
                timeout=timeout,
                gc_after_trial=True,
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            print("\n🛑 用户手动中断优化。")

        self._summarize_results()
        self._generate_visualizations()
        
        return self.study
        
    # _summarize_results 和 _generate_visualizations 方法保持不变...
    def _summarize_results(self):
        """打印和保存最终优化结果."""
        print("\n" + "="*60)
        print("🎉 超参数优化完成！")
        
        try:
            print(f"  - 完成的试验次数: {len(self.study.trials)}")
            print(f"  - 最佳试验 Trial: #{self.study.best_trial.number}")
            print(f"  - 最佳验证 MAE: {self.study.best_value:.6f}")
            print("  - 最佳参数:")
            for key, value in self.study.best_params.items():
                print(f"    - {key}: {value}")
        except ValueError:
            print("  - 没有成功完成的试验。")
        
        print("="*60)

    def _generate_visualizations(self):
        """生成并保存可视化图表."""
        if not self.study.trials:
            return
        
        vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        print(f"\n📊 正在生成可视化图表，将保存至: {vis_dir}")

        try:
            vis.plot_optimization_history(self.study).write_html(os.path.join(vis_dir, "history.html"))
            vis.plot_param_importances(self.study).write_html(os.path.join(vis_dir, "importances.html"))
            vis.plot_parallel_coordinate(self.study).write_html(os.path.join(vis_dir, "parallel_coordinate.html"))
            vis.plot_slice(self.study).write_html(os.path.join(vis_dir, "slice.html"))
            print("✅ 可视化图表生成完毕。")
        except (ValueError, ZeroDivisionError) as e:
            print(f"⚠️ 部分可视化图表生成失败（可能是试验次数不足）: {e}")