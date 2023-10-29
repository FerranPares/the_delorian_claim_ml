# Copyright (c) OpenMMLab. All rights reserved.
import os

from mmengine.config import Config
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


def main():
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = "0"
        os.environ['RANK'] = "0"
        
    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile("configs/car_damage/small_DCN_plus_cfg.py")
    cfg.launcher = "none"
    cfg.work_dir = "./small_work_dir"


    # resume is determined in this priority: resume from > auto_resume
    cfg.resume = True
    cfg.load_from = None

    # build the default runner
    runner = Runner.from_cfg(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
