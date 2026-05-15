#!/usr/bin/env python 
# -*- coding: utf-8 -*-
'''
@Time : 2021-12-20 21:16
@Author : Xinkai Ji
@contact: jhjxk@hotmail.com
@File : video_inference_main.py 
@Software: PyCharm
@desc: 
'''
import os

# WSL + newer torch/cudnn combinations may repeatedly fail cuDNN v8 plan
# selection on legacy models and then fall back anyway. Prefer the legacy
# cuDNN path unless the user explicitly overrides this in the environment.
os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")

from video_inference.video_process import DroneVideoProcess
from video_inference.video_stabilization import DroneVideoStab
from video_inference.video_process_multiprocessing import run
from video_inference.video_det_process_multiprocessing import run as run_det
import argparse
import logging  # 引入logging模块
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s-%(filename)s[%(lineno)d]-%(levelname)s: %(message)s')  # logging.basicConfig函数对日志的输出格式及方式做相关配置

def parse_args():
    parser = argparse.ArgumentParser(description='OpenVTER Implementation')
    parser.add_argument("-c", "--config_json", "--config", dest="config_json", type=str, help='config')
    parser.add_argument("-s","--step",
                        type=int,
                        help="1:stabilize 2:detect video without stabilization 3: detect and tracking video")
    parser.add_argument("-e", "--config_parameter",
                        type=int,
                        help="1:output the stabilize pkl file 2:output stabilize video")
    parser.add_argument("-m", "--multiprocessing",
                        action="store_true", default=False,
                        help="multiprocessing trajectory extraction")
    args = parser.parse_args()
    return args

def run_pipeline(config_path, step, config_parameter=None, multiprocessing=False):
    if step == 1:
        video_stab = DroneVideoStab(config_path)
        video_stab.process(step=config_parameter)
    elif step == 2:
        run_det(config_path)
    elif step == 3:
        if multiprocessing:
            run(config_path)
        else:
            v = DroneVideoProcess(config_path)
            v.process_video()
    else:
        raise ValueError(f"Unsupported step: {step}")


def main():
    args = parse_args()
    run_pipeline(
        config_path=args.config_json,
        step=args.step,
        config_parameter=args.config_parameter,
        multiprocessing=args.multiprocessing,
    )


if __name__ == '__main__':
    main()

