import optuna
from optuna.samplers import TPESampler
import copy
import argparse
from run import main, args_parser  # 导入主函数和参数解析器
from exputils import set_seed  # 导入随机种子设置函数
import json
import os
from datetime import datetime
import time

def run_bayesian_optimization(base_args, n_trials=5):
    """
    执行贝叶斯优化寻找最佳超参数
    :param base_args: 基础配置参数
    :param n_trials: 优化试验次数
    :return: 最佳超参数组合
    """
    def objective(trial):
        # 创建当前试验的参数副本
        trial_args = copy.deepcopy(base_args)

        # 定义超参数搜索空间
        trial_args.lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        trial_args.pos_w = trial.suggest_float('pos_w', 0.1, 1.0)
        trial_args.mani_w = trial.suggest_float('mani_w', 0.01, 0.5)
        trial_args.bs = trial.suggest_categorical('bs', [16, 32, 64, 128])
        trial_args.layer = trial.suggest_categorical('layer', [1,2,3,4])
        trial_args.mlp_layer= trial.suggest_categorical('mlp_layer', [1,2,3,4])
        # 设置随机种子（确保公平比较）
        set_seed(trial_args.random_seed)

        # 运行主训练流程
        trial_args.run_bayesian_optimization = False  # 防止递归调用
        trial_args.checkpoint_path = None  # 不使用预训练模型
        trial_args.get_image = False  # 禁用可视化以加快速度

        # 创建唯一的日志路径（包含 trial.number, timestamp, pid，避免并行冲突）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trial_args.log_dir = os.path.join(base_args.log_dir, f"bayesian_trial_{trial.number}_{timestamp}_{os.getpid()}")
        os.makedirs(trial_args.log_dir, exist_ok=True)

        # 保存当前试验的参数
        with open(os.path.join(trial_args.log_dir, "params.json"), "w") as f:
            json.dump(vars(trial_args), f, indent=4)

        # 运行训练流程并直接获取返回的 best_valid_score
        try:
            valid_score = main(trial_args, optuna_trial=trial)
            # 如果 main 正常返回，valid_score 应该是 float
            try:
                return float(valid_score)
            except:
                return float('inf')
        except optuna.exceptions.TrialPruned:
            # 让 Optuna 记录该 trial 为被剪枝
            raise
        except Exception as e:
            # 如果发生异常，尝试从 metrics.json 读取（如果 main 在失败前写了它）
            metrics_path = os.path.join(trial_args.log_dir, "metrics.json")
            try:
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)
                    if metrics.get("best_valid_score") is not None:
                        return float(metrics["best_valid_score"])
            except Exception:
                pass
            # 否则返回一个很差的分数以示失败（避免 study crash）
            print(f"Trial {trial.number} failed with exception: {e}")
            return float('inf')

    # 创建Optuna study
    sampler = TPESampler(seed=base_args.random_seed)  # 可重复的采样器
    study = optuna.create_study(
        direction='minimize',  # 因为我们要最小化MAE/RMSD
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    # 执行优化
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # 输出最佳结果
    print("\n=== Bayesian Optimization Results ===")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Validation Score: {study.best_trial.value:.6f}")
    print("Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    # 保存最佳参数
    best_params_path = f"{base_args.log_dir}/best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(study.best_trial.params, f, indent=4)
    print(f"Saved best parameters to {best_params_path}")

    return study.best_trial.params

if __name__ == '__main__':
    # 解析基础参数
    base_args = args_parser()

    # 设置贝叶斯优化参数
    base_args.run_bayesian_optimization = True
    base_args.optim_trials = 30  # 试验次数

    # 创建主日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_args.log_dir = f"logs/bayesian_optimization_{timestamp}"
    os.makedirs(base_args.log_dir, exist_ok=True)

    # 运行贝叶斯优化
    best_params = run_bayesian_optimization(base_args, n_trials=base_args.optim_trials)

    # 使用最佳参数训练最终模型
    print("\n=== Training Final Model with Best Parameters ===")
    final_args = copy.deepcopy(base_args)
    final_args.lr = best_params['lr']
    final_args.pos_w = best_params['pos_w']
    final_args.mani_w = best_params['mani_w']
    final_args.bs = best_params['bs']
    final_args.run_bayesian_optimization = False
    final_args.log_dir = f"{base_args.log_dir}/final_model"
    os.makedirs(final_args.log_dir, exist_ok=True)

    # 保存最终参数
    with open(f"{final_args.log_dir}/final_params.json", "w") as f:
        json.dump(vars(final_args), f, indent=4)

    # 运行最终训练
    main(final_args)
