#!/usr/bin/python
# -*- encoding: utf-8 -*-

# 导入必要的库用于时间处理、操作系统交互和时间计算
import datetime
import os
import time
import copy  # 新增：用于深拷贝配置

# 导入PyTorch相关库用于深度学习模型构建和训练
import torch
import wandb  # 用于实验跟踪和可视化
import torch.nn as nn
import torchmetrics  # 用于计算各种评估指标
from torch_geometric.transforms import Compose  # 用于组合多个数据变换

# 新增：导入 Optuna
import optuna
import optuna.exceptions  # 新增：用于剪枝异常处理
from optuna.integration import TensorBoardCallback
import tensorboard

# 导入自定义模块
from model import GCPNet  # GCPNet模型的主要实现
from utils.keras_callbacks import WandbCallback  # Wandb回调函数
from utils.dataset_utils import MP18, dataset_split, get_dataloader  # 数据集处理工具
from utils.flags import Flags  # 配置参数管理
from utils.train_utils import KerasModel, LRScheduler  # 训练工具和学习率调度器
from utils.transforms import GetAngle, ToFloat  # 数据变换工具

from utils.optuna_optimizer import OptunaHyperparameterOptimizer

# 设置NumExpr库的最大线程数为24，用于加速数值计算
os.environ["NUMEXPR_MAX_THREADS"] = "24"
# 开启调试模式，用于打印调试信息
debug = True 

# 导入日志相关库
import logging
from logging.handlers import RotatingFileHandler

# 新增：显存管理工具
import gc
import time

class GPUMemoryManager:
    def __init__(self, verbose=True):
        self.verbose = verbose
    
    def get_memory_info(self):
        if not torch.cuda.is_available():
            return {"total": 0, "allocated": 0, "cached": 0, "free": 0}
        
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        free = total - cached
        
        
        return {"total": total, "allocated": allocated, "cached": cached, "free": free}
    
    def safe_cleanup(self, force=False):
        if not torch.cuda.is_available():
            return
        
        before = self.get_memory_info()
        gc.collect()
        torch.cuda.empty_cache()
        
        if force:
            torch.cuda.synchronize()
            time.sleep(0.1)
            torch.cuda.empty_cache()
        
        after = self.get_memory_info()
        if self.verbose:
            freed = before['cached'] - after['cached']
            if freed > 0.1:
                print(f"🧹 Cleaned {freed:.2f}GB GPU memory")

# 配置日志系统，用于记录训练过程中的重要信息
def log_config(log_file='test.log'):
    # 定义日志格式：[时间戳][日志级别]: 消息内容
    LOG_FORMAT = '[%(asctime)s][%(levelname)s]: %(message)s'
    # 设置日志级别为INFO，记录重要的训练信息
    level = logging.INFO
    # 配置基础日志设置
    logging.basicConfig(level=level, format=LOG_FORMAT)
    # 创建文件日志处理器，支持日志轮转（最大2MB，保留3个备份文件）
    log_file_handler = RotatingFileHandler(filename=log_file, maxBytes=2*1024*1024, backupCount=3)
    # 设置日志格式化器
    formatter = logging.Formatter(LOG_FORMAT)
    log_file_handler.setFormatter(formatter)
    # 将文件处理器添加到根日志记录器
    logging.getLogger('').addHandler(log_file_handler)

# 设置随机种子，确保实验的可重复性
def set_seed(seed):
    # 导入随机数生成相关库
    import random
    import numpy as np
    # 设置Python原生random模块的随机种子
    random.seed(seed)
    # 设置NumPy的随机种子
    np.random.seed(seed)
    # 设置PyTorch CPU操作的随机种子
    torch.manual_seed(seed)
    # 设置PyTorch GPU操作的随机种子（适用于所有GPU）
    torch.cuda.manual_seed_all(seed)
    # 启用确定性算法，确保GPU计算结果可重复
    torch.backends.cudnn.deterministic = True
    # 禁用cudnn的benchmark模式，虽然可能影响性能但保证结果一致
    torch.backends.cudnn.benchmark = False

# 设置和初始化数据集，这是GCPNet训练的第一步
def setup_dataset(config):
    dataset = MP18(root=config.dataset_path, name=config.dataset_name, transform=Compose([GetAngle(), ToFloat(
        )]), r=config.max_edge_distance, n_neighbors=config.n_neighbors, edge_steps=config.edge_input_features, image_selfloop=True, points=config.points, target_name=config.target_name)
    return dataset

# 初始化GCPNet模型，配置模型的各种超参数
def setup_model(dataset, config):
    net = GCPNet(
            data=dataset,
            firstUpdateLayers=config.firstUpdateLayers,
            secondUpdateLayers=config.secondUpdateLayers,
            atom_input_features=config.atom_input_features,
            edge_input_features=config.edge_input_features,
            triplet_input_features=config.triplet_input_features,
            embedding_features=config.embedding_features,
            hidden_features=config.hidden_features,
            output_features=config.output_features,
            min_edge_distance=config.min_edge_distance,
            max_edge_distance=config.max_edge_distance,
            link=config.link,
            dropout_rate=config.dropout_rate,
        )
    return net

# 设置优化器，用于模型参数的更新
def setup_optimizer(net, config):
    optimizer = getattr(torch.optim, config.optimizer)(
        net.parameters(),
        lr=config.lr,
        **config.optimizer_args
    )
    if config.debug:
        print(f"optimizer: {optimizer}")
    return optimizer

# 设置学习率调度器，用于在训练过程中动态调整学习率
def setup_schduler(optimizer, config):
    scheduler = LRScheduler(optimizer, config.scheduler, config.scheduler_args)
    return scheduler

# 构建Keras风格的模型包装器，简化训练流程
def build_keras(net, optimizer, scheduler):
    model = KerasModel(
        net=net,
        loss_fn=nn.L1Loss(),
        metrics_dict={
            "mae": torchmetrics.MeanAbsoluteError(),
            "mape": torchmetrics.MeanAbsolutePercentageError()
        }, 
        optimizer=optimizer,
        lr_scheduler=scheduler
    )
    return model

# 主要的训练函数，执行完整的模型训练流程
def train(config, printnet=False, trial=None):  # 新增trial参数用于剪枝
    name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 如果启用了wandb日志记录，初始化Weights & Biases实验跟踪
    if config.log_enable:
        wandb.init(project=config.project_name, name=name, save_code=False)

    # 第1步：加载和准备数据
    dataset = setup_dataset(config)
    train_dataset, val_dataset, test_dataset = dataset_split(
        dataset, train_size=0.8, valid_size=0.1, test_size=0.1, seed=config.seed, debug=debug) 
    train_loader, val_loader, test_loader = get_dataloader(
        train_dataset, val_dataset, test_dataset, config.batch_size, config.num_workers)

    # 第2步：加载和初始化网络模型
    rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = setup_model(dataset, config).to(rank)
    if config.debug and printnet:
        print(net)

    # 第3步：设置优化器和学习率调度器
    optimizer = setup_optimizer(net, config)
    scheduler = setup_schduler(optimizer, config)

    # 第4步：开始训练过程
    if config.log_enable:
        callbacks = [WandbCallback(project=config.project_name, config=config)]
    else:
        callbacks = None
    
    model = build_keras(net, optimizer, scheduler)
    
    # 开始训练模型（传递trial用于剪枝）
    history = model.fit(
        train_loader,
        val_loader,
        ckpt_path=os.path.join(config.output_dir, config.net+'.pth'),
        epochs=config.epochs,
        monitor='val_loss',
        mode='min',
        patience=config.patience,
        plot=True,
        callbacks=callbacks,
        trial=trial  # 新增：传递trial对象用于剪枝
    )
    
    # 在测试集上评估模型性能
    test_result = model.evaluate(test_loader)
    print(test_result)
    
    # 如果启用日志记录，记录最终的测试结果
    if config.log_enable:
        wandb.log({
            "test_mae": test_result['val_mae'],
            "test_mape": test_result['val_mape'],
            "total_params": model.total_params()
        })
        wandb.finish()

    # 返回训练历史和最佳验证误差
    best_val_mae = min(history['val_mae']) if 'val_mae' in history else float('inf')
    return best_val_mae

def objective(trial, base_config, full_config_from_yaml):
    """
    一个完全由YAML配置文件驱动的Optuna目标函数。
    """
    config = copy.deepcopy(base_config)
    config.seed = base_config.seed + trial.number # 每个trial使用不同的、可复现的种子

    # 1. 从YAML配置中动态读取搜索空间并应用超参数
    search_space = full_config_from_yaml.get('optuna', {}).get('search_space', {})
    if not search_space:
        raise ValueError("配置文件中未找到 'optuna.search_space' 部分!")

    for param, settings in search_space.items():
        suggest_type = settings.get('type')
        if suggest_type == 'categorical':
            value = trial.suggest_categorical(param, settings['choices'])
        elif suggest_type == 'float':
            value = trial.suggest_float(param, settings['low'], settings['high'], log=settings.get('log', False))
        elif suggest_type == 'int':
            value = trial.suggest_int(param, settings['low'], settings['high'], step=settings.get('step', 1))
        else:
            raise ValueError(f"不支持的搜索类型: {suggest_type}")

        # 将建议的值赋给config对象
        if hasattr(config, param):
            setattr(config, param, value)
        elif param == 'weight_decay': # 特殊处理优化器参数
            config.optimizer_args['weight_decay'] = value
        else:
            setattr(config, param, value)

    # 2. 打印、创建目录并开始训练 (此部分逻辑与您的提议一致)
    print(f"\n🚀 Starting Trial #{trial.number}")
    print("  Parameters:")
    for key, value in trial.params.items():
        print(f"    - {key}: {value}")

    trial_name = f"trial_{trial.number}"
    config.output_dir = os.path.join(base_config.output_dir, trial_name)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    try:
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        best_val_mae = train(config, printnet=False, trial=trial)
        print(f"\n🏁 Trial #{trial.number} completed! Result: {best_val_mae:.6f}")
        return best_val_mae
    except optuna.exceptions.TrialPruned:
        print(f"\n✂️  Trial #{trial.number} pruned!")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        raise
    except Exception as e:
        print(f"❌ Trial {trial.number} failed: {e}")
        import traceback
        traceback.print_exc()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return float('inf') # 返回一个很大的值表示失败

# 交叉验证训练函数，用于更可靠的模型性能评估
def train_CV(config):
    from utils.dataset_utils_strong import loader_setup_CV, split_data_CV
    dataset = setup_dataset(config)
    cv_dataset = split_data_CV(dataset, num_folds=config.num_folds, seed=config.seed)
    cv_error = []

    for index in range(0, len(cv_dataset)):
        if not(os.path.exists(output_dir := f"{config.output_dir}/{index}")):
            os.makedirs(output_dir)

        train_loader, test_loader, train_dataset, _ = loader_setup_CV(
            index, config.batch_size, cv_dataset, num_workers=config.num_workers
        )
        
        rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = setup_model(dataset, config).to(rank)
        
        optimizer = setup_optimizer(net, config)
        scheduler = setup_schduler(optimizer, config)

        model = build_keras(net, optimizer, scheduler)
        model.fit(
            train_loader, 
            None,
            ckpt_path=os.path.join(output_dir, config.net+'.pth'), 
            epochs=config.epochs,
            monitor='train_loss',
            mode='min', 
            patience=config.patience, 
            plot=True
        )

        test_error = model.evaluate(test_loader)['val_mae']
        logging.info("fold: {:d}, Test Error: {:.5f}".format(index+1, test_error)) 
        cv_error.append(test_error)
    
    import numpy as np
    mean_error = np.array(cv_error).mean()
    std_error = np.array(cv_error).std()
    logging.info("CV Error: {:.5f}, std Error: {:.5f}".format(mean_error, std_error))
    return cv_error

# 预测函数，用于在新数据上进行推理
def predict(config):
    dataset = setup_dataset(config)
    from torch_geometric.loader import DataLoader
    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=False,)

    rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = setup_model(dataset, config).to(rank)

    optimizer = setup_optimizer(net, config)
    scheduler = setup_schduler(optimizer, config)

    model = build_keras(net, optimizer, scheduler)
    model.predict(test_loader, ckpt_path=config.model_path, test_out_path=config.output_path)

# 可视化函数，用于分析和可视化模型的特征表示
def visualize(config):
    from utils.dataset_utils_strong import MP18, dataset_split, get_dataloader
    from utils.transforms import GetAngle, ToFloat

    dataset = MP18(root=config.dataset_path,name=config.dataset_name,transform=Compose([GetAngle(),ToFloat()]), r=config.max_edge_distance, n_neighbors=config.n_neighbors, edge_steps=config.edge_input_features, image_selfloop=True,points=config.points,target_name=config.target_name)

    train_dataset, val_dataset, test_dataset = dataset_split(dataset,train_size=0.8,valid_size=0.1,test_size=0.1,seed=config.seed, debug=debug)
    train_loader, val_loader, test_loader = get_dataloader(train_dataset, val_dataset, test_dataset, config.batch_size, config.num_workers)

    rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = setup_model(dataset, config).to(rank)

    optimizer = setup_optimizer(net, config)
    scheduler = setup_schduler(optimizer, config)
    print("optimizer:",optimizer)

    model = KerasModel(net=net, loss_fn=nn.L1Loss(), metrics_dict={"mae":torchmetrics.MeanAbsoluteError(),"mape":torchmetrics.MeanAbsolutePercentageError()},optimizer=optimizer,lr_scheduler = scheduler)
    data_loader, _, _ = get_dataloader(dataset, val_dataset, test_dataset, config.batch_size, config.num_workers)
   
    model.analysis(net_name=config.net, test_data=data_loader,ckpt_path=config.model_path,tsne_args=config.visualize_args)

    return model

# 主程序入口点，程序从这里开始执行
# main.py (推荐的最终版本)

if __name__ == "__main__":
    # 忽略特定的警告信息
    import warnings
    warnings.filterwarnings('ignore', '.*TypedStorage is deprecated.*')

    # 1. 创建Flags实例，让它自己完成所有配置和参数的解析
    # 这是唯一需要创建Flags实例的地方
    flags = Flags()
    # 2. 直接访问 updated_config 属性来获取最终配置，而不是调用它
    config = flags.updated_config

    # 从已经合并好的config对象中获取任务类型
    task_type = config.task_type

    # 主输出目录，所有任务都在此目录下创建子文件夹
    main_output_dir = config.output_dir
    os.makedirs(main_output_dir, exist_ok=True)
    
if __name__ == "__main__":
    # 忽略特定的警告信息
    import warnings
    warnings.filterwarnings('ignore', '.*TypedStorage is deprecated.*')
    
    import yaml
    
    # 1. 直接创建Flags实例，让它自己完成所有命令行参数的解析
    flags = Flags()
    
    # 2. 从Flags实例的属性中获取最终的、合并好的配置对象
    config = flags.updated_config
    
    # 3. 从config对象中安全地获取task_type，因为它已经被Flags处理好了
    task_type = config.task_type
    
    # 主输出目录，所有任务都在此目录下创建子文件夹
    main_output_dir = config.output_dir
    os.makedirs(main_output_dir, exist_ok=True)
    
    # --- 根据任务类型进行分支处理 (这部分逻辑不变) ---
    if task_type.lower() == 'train':
        name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        config.output_dir = os.path.join(main_output_dir, f"train_{name}")
        os.makedirs(config.output_dir, exist_ok=True)
        set_seed(config.seed)
        print(f"📦 单次训练任务，输出至: {config.output_dir}")
        train(config)
            
    elif task_type.lower() == 'hyperparameter':
        print(f"🗄️ 超参数优化任务，主输出目录: {main_output_dir}")
        
        with open(config.config_file, 'r') as f:
            full_config = yaml.safe_load(f)
        
        from utils.optuna_optimizer import OptunaHyperparameterOptimizer
        
        optimizer = OptunaHyperparameterOptimizer(config, train, full_config)
        study = optimizer.run()
        
    elif task_type.lower() == 'cv':
        name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        config.output_dir = os.path.join(main_output_dir, f"cv_{name}")
        os.makedirs(config.output_dir, exist_ok=True)
        print(f"📦 交叉验证任务，输出至: {config.output_dir}")
        log_file = os.path.join(config.output_dir, config.project_name + '.log')
        log_config(log_file)
        train_CV(config)
        
    elif task_type.lower() == 'predict':
        name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        config.output_dir = os.path.join(main_output_dir, f"predict_{name}")
        os.makedirs(config.output_dir, exist_ok=True)
        print(f"📦 预测任务，输出至: {config.output_dir}")
        predict(config)
        
    elif task_type.lower() == 'visualize':
        print(f"🎨 可视化任务...")
        visualize(config)
        
    else:
        raise NotImplementedError(f"不支持的任务类型: {task_type}")