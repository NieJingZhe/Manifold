#!/usr/bin/env python3
"""Entry point for training (modularized).
Usage: python train_refactor.py
Assumes project-provided modules: args_parse, exputils, dataset, models, utils.*
"""
import os
import logging
from torch.utils.tensorboard import SummaryWriter
from args_parse import args_parser
from exputils import initialize_exp, set_seed, get_dump_path, merge_args_from_paths
from trainer import Trainer
from utils.helpers import get_arg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def main(args=None, optuna_trial=None):
    #变量可以去查看args_parser里面的说明，pos就是坐标，mani是manifold的缩写目前maniloss采用的是CE
    #然后models跟manifold文件夹中可能有不少之前的废稿
    if args is None:
        args = args_parser()
    try:
        args = merge_args_from_paths(args, attr_name="paras_path")
    except Exception:
        logger.exception("merge_args_from_paths 出错（继续使用原始 args）")

    if get_arg(args, "run_bayesian_optimization", False):
        logger.info("run_bayesian_optimization 被启用 -> 请从外部调用对应流程")
        return

    # init experiment
    initialize_exp(args)
    set_seed(getattr(args, "random_seed", 0))
    logger_path = get_dump_path(args)
    writer = SummaryWriter(log_dir=os.path.join(logger_path, "tensorboard"))

    trainer = Trainer(args, writer, logger_path)
    try:
        result = trainer.run(optuna_trial=optuna_trial)
    finally:
        try:
            writer.close()
        except Exception:
            pass

    return result

if __name__ == '__main__':
    main()
