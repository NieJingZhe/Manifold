# utils/optuna.py
"""
集中管理训练过程中的 metrics 写入、Optuna 报告与异常记录逻辑。
将复杂的 IO/异常逻辑从 Trainer 中抽离，保持 Trainer 主流程简洁。
"""
#用来处理贝叶斯优化的东西
import os
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# 我们复用项目里已有的原子写入函数（如果你已经在 utils.helpers 中有 _atomic_write_json）
try:
    from .helpers import _atomic_write_json
except Exception:
    # 如果没有 helpers 的 _atomic_write_json，则提供一个简单的备选实现
    import json
    import tempfile

    def _atomic_write_json(path, obj):
        d = os.path.dirname(path)
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)


def report_optuna_and_maybe_prune(optuna_trial, best_valid_score: float, epoch: int, logger_path: str):
    """
    如果 optuna_trial 非空，则上报当前 best_valid_score 并检查是否需要 prune。
    若需要 prune，会在 logger_path 写入 metrics.json 并抛出 optuna.exceptions.TrialPruned。
    任何异常都会被捕获并记录（不阻塞训练主流程）。
    """
    if optuna_trial is None:
        return

    try:
        # 与 Trainer 原行为保持一致：如果 best_valid_score 仍为 inf，向 optuna 上报 None
        to_report = None if best_valid_score == float("inf") else best_valid_score
        optuna_trial.report(to_report, epoch)
        if optuna_trial.should_prune():
            metrics = {
                "status": "pruned",
                "best_valid_score": None if best_valid_score == float("inf") else best_valid_score,
                "best_test_score": None,  # test 未必计算
                "best_epoch": None,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            try:
                _atomic_write_json(os.path.join(logger_path, "metrics.json"), metrics)
            except Exception:
                logger.exception("写入 pruned metrics 失败（忽略）")
            # 抛出 optuna 的剪枝异常（调用方应处理或让其继续向上抛）
            raise optuna.exceptions.TrialPruned()
    except Exception:
        logger.exception("Optuna 报告失败（忽略）")


def write_final_metrics(logger_path: str, best_valid_score: float, best_test_score: float, best_epoch: Optional[int]):
    """
    在训练正常结束时写入 final metrics（status='finished'）。
    使用 _atomic_write_json 保证原子写入；异常会记录但不抛出。
    """
    metrics = {
        "status": "finished",
        "best_valid_score": None if best_valid_score == float("inf") else best_valid_score,
        "best_test_score": None if best_test_score == float("inf") else best_test_score,
        "best_epoch": best_epoch,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        _atomic_write_json(os.path.join(logger_path, "metrics.json"), metrics)
    except Exception:
        logger.exception("写入 final metrics 失败")


def handle_training_exception(logger_path: str, exc: Exception, best_valid_score: float, best_test_score: float, best_epoch: Optional[int]):
    """
    集中化异常记录：训练过程中发生未捕获异常时调用。
    - 会写入 status='failed' 的 metrics.json，包含错误信息与当前 best 状态。
    - 如果写入失败，会做最小化的 fallback（写文本），以避免在异常处理器中再次抛错。
    """
    try:
        metrics = {
            "status": "failed",
            "error": repr(exc),
            "best_valid_score": None if best_valid_score == float("inf") else best_valid_score,
            "best_test_score": None if best_test_score == float("inf") else best_test_score,
            "best_epoch": best_epoch,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            _atomic_write_json(os.path.join(logger_path, "metrics.json"), metrics)
        except Exception:
            logger.exception("写入 failed metrics 失败，尝试 fallback 文本写入")
            # fallback
            os.makedirs(logger_path, exist_ok=True)
            with open(os.path.join(logger_path, "metrics_failed.txt"), "w", encoding="utf-8") as f:
                f.write("Failed to write metrics.json in exception handler.\n")
                f.write("Original exception: " + repr(exc) + "\n")
    except Exception:
        # 绝对兜底：不能再 raise
        try:
            os.makedirs(logger_path, exist_ok=True)
            with open(os.path.join(logger_path, "metrics_failed_2.txt"), "w", encoding="utf-8") as f:
                f.write("Exception when handling training exception. Last-resort dump.\n")
        except Exception:
            pass
