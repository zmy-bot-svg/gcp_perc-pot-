2025-07-12 20:49:09.593370: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-07-12 20:49:09.607423: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1752324549.629117    2235 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1752324549.633858    2235 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1752324549.646757    2235 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1752324549.646772    2235 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1752324549.646773    2235 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1752324549.646774    2235 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2025-07-12 20:49:09.650363: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
[I 2025-07-12 20:49:11,887] Using an existing study with name 'GCPNet_PotNet_4090_hyperopt' instead of creating a new one.
🗄️ 超参数优化任务，主输出目录: ./output_potnet_4090
💾 存储路径 (Storage): sqlite:///./output_potnet_4090/GCPNet_PotNet_4090_hyperopt.db
✂️ 剪枝器 (Pruner): 已从配置文件加载 MedianPruner (参数: {'n_startup_trials': 5, 'n_warmup_steps': 5, 'interval_steps': 2})
🔭 采样器 (Sampler): 已从配置文件加载 TPESampler (参数: {'n_startup_trials': 10})
📈 研究已加载，历史试验次数: 1

🚀 开始优化... 将运行 80 次试验。
  0%|          | 0/80 [00:00<?, ?it/s]                                        0%|          | 0/80 [00:00<?, ?it/s]                                        0%|          | 0/80 [00:00<?, ?it/s]  0%|          | 0/80 [00:00<?, ?it/s]
[W 2025-07-12 20:49:11,936] Trial 1 failed with parameters: {'lr': 0.0005295232182223216, 'dropout_rate': 0.21062160831870796, 'weight_decay': 9.951114114484866e-05} because of the following error: ValueError('CategoricalDistribution does not support dynamic value space.').
Traceback (most recent call last):
  File "/root/miniconda3/envs/gcp_final_env/lib/python3.10/site-packages/optuna/study/_optimize.py", line 201, in _run_trial
    value_or_values = func(trial)
  File "/root/autodl-tmp/gcp_pot_jian/utils/optuna_optimizer.py", line 119, in _objective
    value = trial.suggest_categorical(param, settings['choices'])
  File "/root/miniconda3/envs/gcp_final_env/lib/python3.10/site-packages/optuna/trial/_trial.py", line 406, in suggest_categorical
    return self._suggest(name, CategoricalDistribution(choices=choices))
  File "/root/miniconda3/envs/gcp_final_env/lib/python3.10/site-packages/optuna/trial/_trial.py", line 641, in _suggest
    storage.set_trial_param(trial_id, name, param_value_in_internal_repr, distribution)
  File "/root/miniconda3/envs/gcp_final_env/lib/python3.10/site-packages/optuna/storages/_cached_storage.py", line 169, in set_trial_param
    self._backend.set_trial_param(trial_id, param_name, param_value_internal, distribution)
  File "/root/miniconda3/envs/gcp_final_env/lib/python3.10/site-packages/optuna/storages/_rdb/storage.py", line 586, in set_trial_param
    self._set_trial_param_without_commit(
  File "/root/miniconda3/envs/gcp_final_env/lib/python3.10/site-packages/optuna/storages/_rdb/storage.py", line 608, in _set_trial_param_without_commit
    trial_param.check_and_add(session, trial.study_id)
  File "/root/miniconda3/envs/gcp_final_env/lib/python3.10/site-packages/optuna/storages/_rdb/models.py", line 353, in check_and_add
    self._check_compatibility_with_previous_trial_param_distributions(session, study_id)
  File "/root/miniconda3/envs/gcp_final_env/lib/python3.10/site-packages/optuna/storages/_rdb/models.py", line 367, in _check_compatibility_with_previous_trial_param_distributions
    distributions.check_distribution_compatibility(
  File "/root/miniconda3/envs/gcp_final_env/lib/python3.10/site-packages/optuna/distributions.py", line 672, in check_distribution_compatibility
    raise ValueError(
ValueError: CategoricalDistribution does not support dynamic value space.
[W 2025-07-12 20:49:11,937] Trial 1 failed with value None.
Traceback (most recent call last):
  File "/root/autodl-tmp/gcp_pot_jian/main.py", line 433, in <module>
    study = optimizer.run()
  File "/root/autodl-tmp/gcp_pot_jian/utils/optuna_optimizer.py", line 179, in run
    self.study.optimize(
  File "/root/miniconda3/envs/gcp_final_env/lib/python3.10/site-packages/optuna/study/study.py", line 489, in optimize
    _optimize(
  File "/root/miniconda3/envs/gcp_final_env/lib/python3.10/site-packages/optuna/study/_optimize.py", line 64, in _optimize
    _optimize_sequential(
  File "/root/miniconda3/envs/gcp_final_env/lib/python3.10/site-packages/optuna/study/_optimize.py", line 161, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
  File "/root/miniconda3/envs/gcp_final_env/lib/python3.10/site-packages/optuna/study/_optimize.py", line 253, in _run_trial
    raise func_err
  File "/root/miniconda3/envs/gcp_final_env/lib/python3.10/site-packages/optuna/study/_optimize.py", line 201, in _run_trial
    value_or_values = func(trial)
  File "/root/autodl-tmp/gcp_pot_jian/utils/optuna_optimizer.py", line 119, in _objective
    value = trial.suggest_categorical(param, settings['choices'])
  File "/root/miniconda3/envs/gcp_final_env/lib/python3.10/site-packages/optuna/trial/_trial.py", line 406, in suggest_categorical
    return self._suggest(name, CategoricalDistribution(choices=choices))
  File "/root/miniconda3/envs/gcp_final_env/lib/python3.10/site-packages/optuna/trial/_trial.py", line 641, in _suggest
    storage.set_trial_param(trial_id, name, param_value_in_internal_repr, distribution)
  File "/root/miniconda3/envs/gcp_final_env/lib/python3.10/site-packages/optuna/storages/_cached_storage.py", line 169, in set_trial_param
    self._backend.set_trial_param(trial_id, param_name, param_value_internal, distribution)
  File "/root/miniconda3/envs/gcp_final_env/lib/python3.10/site-packages/optuna/storages/_rdb/storage.py", line 586, in set_trial_param
    self._set_trial_param_without_commit(
  File "/root/miniconda3/envs/gcp_final_env/lib/python3.10/site-packages/optuna/storages/_rdb/storage.py", line 608, in _set_trial_param_without_commit
    trial_param.check_and_add(session, trial.study_id)
  File "/root/miniconda3/envs/gcp_final_env/lib/python3.10/site-packages/optuna/storages/_rdb/models.py", line 353, in check_and_add
    self._check_compatibility_with_previous_trial_param_distributions(session, study_id)
  File "/root/miniconda3/envs/gcp_final_env/lib/python3.10/site-packages/optuna/storages/_rdb/models.py", line 367, in _check_compatibility_with_previous_trial_param_distributions
    distributions.check_distribution_compatibility(
  File "/root/miniconda3/envs/gcp_final_env/lib/python3.10/site-packages/optuna/distributions.py", line 672, in check_distribution_compatibility
    raise ValueError(
ValueError: CategoricalDistribution does not support dynamic value space.
