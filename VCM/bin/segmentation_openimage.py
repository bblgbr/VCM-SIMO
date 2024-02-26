#!/usr/bin/env python3
import argparse
import glob
import json
import os


# os.environ["DETECTRON2_DATASETS"] = './dataset/validation_2'
import sys
import utils.simo_utils as simo_utils
from utils.eval_vivo_mask import DetectEval
import time
import torch



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index", default=31, type=int)
    parser.add_argument("-n", "--number", default=5000, type=int)
    parser.add_argument("-m", "--mode", default='feature_coding')
    parser.add_argument("-b", "--bpp-savepath", default='../result/output/default.json')
    parser.add_argument("-p", "--pick", default=1, type=int)
    parser.add_argument("--method", default='Cheng')
    parser.add_argument("--bin-paths", default='./streams_SIMO/openimg_segmen_plyr_anch2_enc_nosave')

    args = parser.parse_args()
    set_idx = args.index
    number = args.number
    mode = args.mode
    bpp_savepath = args.bpp_savepath
    method = args.method
    bin_paths = args.bin_paths
    torch.set_default_dtype(torch.float64)


    with open(os.path.join(os.environ["SETTING_DIR"], f"{set_idx}.json"), "r") as setting_json:
        settings = json.load(setting_json)  # setting 中是model_name, yaml_path和pkl_path

    if settings["model_name"] == "x101":
        if method == 'Cheng':
            methods_eval = DetectEval(settings, set_idx, bpp_savepath)
        picklist = sorted(glob.glob(os.path.join(os.environ["DETECTRON2_DATASETS"], "*.jpg")))[:number]
        picklist = [simo_utils.simple_filename(x) for x in picklist]
        methods_eval.prepare_part(picklist, data_name=f"pick{args.pick}")

    start_proc_time = time.process_time()
    start_perf_time = time.perf_counter()

    if mode == "feature_encode":
        methods_eval.feature_encode(bin_paths)
    elif mode == "feature_decode":
        methods_eval.feature_decode(bin_paths)
    elif mode == "summary":
        methods_eval.summary()

    inference_proc_time = time.process_time() - start_proc_time
    inference_perf_time = time.perf_counter() - start_perf_time
    print(f'Inference time: Process: {inference_proc_time}  Perf ctr: {inference_perf_time}')

