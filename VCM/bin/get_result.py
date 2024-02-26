#!/usr/bin/env python3

import argparse
import subprocess
import os
import pandas as pd
import csv
import json
from utils.get_result import EVAL_mAP

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index", default=0, type=int)
    parser.add_argument("-m", "--mode", default='det')
    args = parser.parse_args()
    mode = args.mode
    index = args.index

    temp_eval = EVAL_mAP(index, mode)
    # temp_eval.forward()
    temp_eval.write_result()

    



    