import os
import sys

import torch
import torch.nn as nn
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog
from tqdm import tqdm
import cv2
import numpy as np
import torch.nn.functional as F
import subprocess

import utils.simo_utils as simo_utils
from utils.quantizer import quant_fix, dequant_fix#
from utils.cvt_detectron_coco_oid_vivo import conversion
import scipy.io as sio
from typing import Tuple, Union
import PIL.Image as Image
import math
import json
import scipy.io as sio
import utils.oid_mask_encoding as oid_mask_encoding
from compressai import set_entropy_coder
import time
from utils.simo_utils import FeatureDump
import detectron2
import compressai
from utils.get_result import EVAL_mAP

def Pfeature_replicatepad(feat, factor=16): #输入feat为[b, 256, h, w]
    h = feat.size()[2]
    w = feat.size()[3]
    if h % factor == 0:
        h_new = h
    else:
        h_new = ((h // factor) + 1) * factor
    if w % factor == 0:
        w_new = w
    else:
        w_new = ((w // factor) + 1) * factor
    h_new_left = (h_new - h) // 2
    h_new_right = (h_new - h) - h_new_left
    w_new_left = (w_new - w) // 2
    w_new_right = (w_new - w) - w_new_left
    # nn.ReplicationPad2d((1, 2, 3, 2))  #左侧填充1行，右侧填充2行，上方填充3行，下方填充2行
    pad_func = nn.ReplicationPad2d((w_new_left, w_new_right, h_new_left, h_new_right))
    feat_pad = pad_func(feat)
    return feat_pad, h_new_left, h_new_right, w_new_left, w_new_right #恢复时h_new_left:(h_now-h_right)

def Pfeature_replicatepad_reverse(feat, h_new_left, h_new_right, w_new_left, w_new_right): #输入feat为[b, 256, h, w]
    h = feat.size()[2]
    w = feat.size()[3]
    feat_new = feat[:, :, h_new_left:(h-h_new_right), w_new_left:(w-w_new_right)]
    return feat_new

def Pfeature_replicatepad_youxiajiao(feat, factor=16): #相比于Pfeature_replicatepad的区别为pad从上下左右变为右下角 输入feat为[b, 256, h, w]
    h = feat.size()[2]
    w = feat.size()[3]
    if h % factor == 0:
        h_new = h
    else:
        h_new = ((h // factor) + 1) * factor
    if w % factor == 0:
        w_new = w
    else:
        w_new = ((w // factor) + 1) * factor
    h_new_left = 0 #(h_new - h) // 2
    h_new_right = h_new - h
    w_new_left = 0
    w_new_right = w_new - w
    # nn.ReplicationPad2d((1, 2, 3, 2))  #左侧填充1行，右侧填充2行，上方填充3行，下方填充2行
    pad_func = nn.ReplicationPad2d((w_new_left, w_new_right, h_new_left, h_new_right))
    feat_pad = pad_func(feat)
    return feat_pad, h_new_left, h_new_right, w_new_left, w_new_right #恢复时h_new_left:(h_now-h_right)

def Pfeature_zeropad_youxiajiao128(feat, factor=16): #相比于Pfeature_replicatepad的区别为pad从上下左右变为右下角 输入feat为[b, 256, h, w]
    h = feat.size()[2]
    w = feat.size()[3]
    if h % factor == 0:
        h_new = h
    else:
        h_new = ((h // factor) + 1) * factor
    if w % factor == 0:
        w_new = w
    else:
        w_new = ((w // factor) + 1) * factor
    #加了下面4行让hw_new最小为128
    if h_new < 128:
        h_new = 128
    if w_new < 128:
        w_new = 128
    h_new_left = 0 #(h_new - h) // 2
    h_new_right = h_new - h
    w_new_left = 0
    w_new_right = w_new - w
    # nn.ReplicationPad2d((1, 2, 3, 2))  #左侧填充1行，右侧填充2行，上方填充3行，下方填充2行
    pad_func = nn.ZeroPad2d((w_new_left, w_new_right, h_new_left, h_new_right))
    feat_pad = pad_func(feat)
    return feat_pad, h_new_left, h_new_right, w_new_left, w_new_right #恢复时h_new_left:(h_now-h_right)

def Pfeature_zeropad_youxiajiao128_reverse(feat, h_new_left, h_new_right, w_new_left, w_new_right): #输入feat为[b, 256, h, w]
    h = feat.size()[2]
    w = feat.size()[3]
    feat_new = feat[:, :, h_new_left:(h-h_new_right), w_new_left:(w-w_new_right)]
    return feat_new

def Pfeature_zeropad_youxiajiao256(feat, factor=16): #相比于Pfeature_replicatepad的区别为pad从上下左右变为右下角 输入feat为[b, 256, h, w]
    h = feat.size()[2]
    w = feat.size()[3]
    if h % factor == 0:
        h_new = h
    else:
        h_new = ((h // factor) + 1) * factor
    if w % factor == 0:
        w_new = w
    else:
        w_new = ((w // factor) + 1) * factor
    #加了下面4行让hw_new最小为128
    if h_new < 256:
        h_new = 256
    if w_new < 256:
        w_new = 256
    h_new_left = 0 #(h_new - h) // 2
    h_new_right = h_new - h
    w_new_left = 0
    w_new_right = w_new - w
    # nn.ReplicationPad2d((1, 2, 3, 2))  #左侧填充1行，右侧填充2行，上方填充3行，下方填充2行
    pad_func = nn.ZeroPad2d((w_new_left, w_new_right, h_new_left, h_new_right))
    feat_pad = pad_func(feat)
    return feat_pad, h_new_left, h_new_right, w_new_left, w_new_right #恢复时h_new_left:(h_now-h_right)

def Pfeature_zeropad_youxiajiao256_reverse(feat, h_new_left, h_new_right, w_new_left, w_new_right): #输入feat为[b, 256, h, w]
    h = feat.size()[2]
    w = feat.size()[3]
    feat_new = feat[:, :, h_new_left:(h-h_new_right), w_new_left:(w-w_new_right)]
    return feat_new

def Pfeature_zeropad_youxiajiao64(feat, factor=16): #相比于Pfeature_replicatepad的区别为pad从上下左右变为右下角 输入feat为[b, 256, h, w]
    h = feat.size()[2]
    w = feat.size()[3]
    if h % factor == 0:
        h_new = h
    else:
        h_new = ((h // factor) + 1) * factor
    if w % factor == 0:
        w_new = w
    else:
        w_new = ((w // factor) + 1) * factor
    #加了下面4行让hw_new最小为64
    if h_new < 64:
        h_new = 64
    if w_new < 64:
        w_new = 64
    h_new_left = 0 #(h_new - h) // 2
    h_new_right = h_new - h
    w_new_left = 0
    w_new_right = w_new - w
    # nn.ReplicationPad2d((1, 2, 3, 2))  #左侧填充1行，右侧填充2行，上方填充3行，下方填充2行
    pad_func = nn.ZeroPad2d((w_new_left, w_new_right, h_new_left, h_new_right))
    feat_pad = pad_func(feat)
    return feat_pad, h_new_left, h_new_right, w_new_left, w_new_right #恢复时h_new_left:(h_now-h_right)

def Pfeature_zeropad_youxiajiao64_reverse(feat, h_new_left, h_new_right, w_new_left, w_new_right): #输入feat为[b, 256, h, w]
    h = feat.size()[2]
    w = feat.size()[3]
    feat_new = feat[:, :, h_new_left:(h-h_new_right), w_new_left:(w-w_new_right)]
    return feat_new

def Pfeature_zeropad_youxiajiao32(feat, factor=16):
    h = feat.size()[2]
    w = feat.size()[3]
    if h % factor == 0:
        h_new = h
    else:
        h_new = ((h // factor) + 1) * factor
    if w % factor == 0:
        w_new = w
    else:
        w_new = ((w // factor) + 1) * factor
    if h_new < 32:
        h_new = 32
    if w_new < 32:
        w_new = 32
    h_new_left = 0 #(h_new - h) // 2
    h_new_right = h_new - h
    w_new_left = 0
    w_new_right = w_new - w
    # nn.ReplicationPad2d((1, 2, 3, 2))  #左侧填充1行，右侧填充2行，上方填充3行，下方填充2行
    pad_func = nn.ZeroPad2d((w_new_left, w_new_right, h_new_left, h_new_right))
    feat_pad = pad_func(feat)
    return feat_pad, h_new_left, h_new_right, w_new_left, w_new_right #恢复时h_new_left:(h_now-h_right)


def Pfeature_zeropad_youxiajiao32_reverse(feat, h_new_left, h_new_right, w_new_left, w_new_right): #输入feat为[b, 256, h, w]
    h = feat.size()[2]
    w = feat.size()[3]
    feat_new = feat[:, :, h_new_left:(h-h_new_right), w_new_left:(w-w_new_right)]
    return feat_new

def padding_size(ori_size, factor_size):
    if ori_size % factor_size == 0:
        return ori_size
    else:
        return factor_size * (ori_size // factor_size + 1)


def mse2psnr(mse):
    # 根据Hyper论文中的内容，将MSE->psnr(db)
    # return 10*math.log10(255*255/mse)
    return 10 * math.log10(1 / mse)


def compute_metrics(
        a: Union[np.array, Image.Image],
        b: Union[np.array, Image.Image],
        max_val: float = 255.0,
) -> Tuple[float, float]:
    """Returns PSNR and MS-SSIM between images `a` and `b`. """
    if isinstance(a, Image.Image):
        a = np.asarray(a)
    if isinstance(b, Image.Image):
        b = np.asarray(b)

    a = torch.from_numpy(a.copy()).float().unsqueeze(0)
    if a.size(3) == 3:
        a = a.permute(0, 3, 1, 2)
    b = torch.from_numpy(b.copy()).float().unsqueeze(0)
    if b.size(3) == 3:
        b = b.permute(0, 3, 1, 2)

    mse = torch.mean((a - b) ** 2).item()
    p = 20 * np.log10(max_val) - 10 * np.log10(mse)
    # m = ms_ssim(a, b, data_range=max_val).item()
    m = 0
    return p, m


class RateDistortionLoss(nn.Module):  # 只注释掉了109行的bpp_loss, 08021808又加上了
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    # def forward(self, output, target, lq, x_l, x_enh): #0.001
    def forward(self, output, target, height, width, flag):  # 0.001 #, lq, x_l, x_enh
        # N, _, _, _ = target.size()
        N, _, H, W = target.size()
        out = {}
        num_pixels_feature = N * H * W
        num_pixels = N * height * width
        # print('ratedistortion functions: image hxw: %dx%d, num_pixel: %d' % (height, width, num_pixels))

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        bpp_temp = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels_feature))
            for likelihoods in output["likelihoods"].values()
        )
        # print('ratedistortion functions: bpp_img/bpp_feat: %8.4f/%8.4f' % (out["bpp_loss"].item(), bpp_temp))
        
        if flag == 3:
            out["mse_loss"] = self.mse(output["x_hat"], target)
        elif flag == 2:
            out["mse_loss"] = self.mse(output["x_hat_p2"], target)
        elif flag == 4:
            out["mse_loss"] = self.mse(output["x_hat_p4"], target)
        elif flag == 5:
            out["mse_loss"] = self.mse(output["x_hat_p5"], target)
        
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        return out




class Eval:
    def __init__(self, settings, index, bpp_savepath) -> None:
        self.settings = settings
        self.set_idx = index
        self.VTM_param = settings["VTM"]
        print('load model path: %s' % (settings["pkl_path"]))
        self.model, self.cfg = simo_utils.model_loader(settings)  # load模型进来
        self.model = self.model.to(os.environ["DEVICE"])
        self.prepare_dir()
        simo_utils.print_settings(settings, index)

        self.pixel_num = settings["pixel_num"]

        compressai_lmbda = 1.0
        self.criterion_p3 = RateDistortionLoss(lmbda=compressai_lmbda)
        self.criterion_p2 = RateDistortionLoss(lmbda=compressai_lmbda)
        self.criterion_p4 = RateDistortionLoss(lmbda=compressai_lmbda)
        self.criterion_p5 = RateDistortionLoss(lmbda=compressai_lmbda)

        path_save = os.path.join(os.environ["VCM_TESTDATA"], 'annotations_5k/Openimage_numpixel_test5000_new.json')  # new_dict[fname_simple][0] [1] [2] 分别为height, width, num_pixel fname_simple为 '000a1249af2bc5f0'
        tf = open(path_save, "r")
        self.numpixel_test5000 = json.load(tf)
        self.path_bppsave = bpp_savepath 
        self.bpp_test5000 = {}
        self.dump = FeatureDump(subsample=9)
        self.det_eval = EVAL_mAP(index, 'det')

    def prepare_dir(self):
        os.makedirs(os.path.join(os.environ["OUTPUT_DIR"], "info"), exist_ok=True)
        os.makedirs(os.path.join(os.environ["OUTPUT_DIR"], "output"), exist_ok=True)
        os.makedirs(os.environ["BITSTERM_DIR"], exist_ok=True)
        for i in range(1, 7):
            os.makedirs(os.path.join(os.environ["BITSTERM_DIR"], f"{i}"), exist_ok=True)
        

    def forward_front(self, inputs, images, features):
        proposals, _ = self.model.proposal_generator(images, features, None)
        results, _ = self.model.roi_heads(images, features, proposals, None)
        return self.model._postprocess(results, inputs, images.image_sizes)

    def summary(self):
        result_csv_path = os.path.join(os.environ['OUTPUT_DIR'], 'results.csv')
        AP_path = os.path.join(os.environ['OUTPUT_DIR'], f'info/{self.set_idx}_AP.txt')
        with open(result_csv_path, "a") as result_f:
            # with open(f"inference/{self.set_idx}_AP.txt", "rt") as ap_f:
            with open(AP_path, "rt") as ap_f:
                ap = ap_f.readline()
                ap = ap.split(",")[1][:-1]
            # ml add
            size_coeffs, size_mean, self.qp, self.DeepCABAC_qstep = 0, 0, 0, 0
            # bpp = (size_basis + size_coeffs, + size_mean)/self.pixel_num
            # ml
            bpp = 0

            print(">>>> result: ", f"{self.set_idx},{self.qp},{self.DeepCABAC_qstep},{bpp},{ap}\n")

            result_f.write(f"{self.set_idx},{self.qp},{self.DeepCABAC_qstep},{bpp},{ap}\n")


    def clear(self):
        DatasetCatalog._REGISTERED.clear()

    def feature_encode(self, bin_paths):
        with tqdm(total=len(self.data_loader.dataset)) as pbar:
            for inputs in iter(self.data_loader):
                start = time.time()
                fname_temp = simo_utils.simple_filename(inputs[0]["file_name"])
                bin_path = os.path.join(bin_paths, fname_temp) + '.bin'
                bin_path_y = bin_path[:-4] + '_y' + bin_path[-4:]
                bin_path_z = bin_path[:-4] + '_z' + bin_path[-4:]
                if os.path.exists(bin_path_y) and os.path.exists(bin_path_z):
                    print(f"{fname_temp} has encoded, skip it")
                    pbar.update()
                    continue
                images = self.model.preprocess_image(inputs)
                self.height_temp = self.numpixel_test5000[fname_temp][0]
                self.width_temp = self.numpixel_test5000[fname_temp][1]
                self.numpixel_temp = self.numpixel_test5000[fname_temp][2]
                
                features = self.model.backbone(images.tensor)

                d = features['p2']  # [1, 256, 200, 304]
                d_p3 = features['p3']
                d_p4 = features['p4']
                d_p5 = features['p5']
                d_originalsize_p2 = d
                d_originalsize_p3 = d_p3
                d_originalsize_p4 = d_p4
                d_originalsize_p5 = d_p5
                # print(d.size(), '-------------------P2 original size')

                if torch.min(d) >= torch.min(d_p3): #2个数中取小的
                    guiyihua_min_1 = torch.min(d_p3)
                else:
                    guiyihua_min_1 = torch.min(d)
                if torch.max(d) >= torch.max(d_p3): #2个数中取大的
                    guiyihua_max_1 = torch.max(d)
                else:
                    guiyihua_max_1 = torch.max(d_p3)

                if torch.min(d_p4) >= torch.min(d_p5): #2个数中取小的
                    guiyihua_min_2 = torch.min(d_p5)
                else:
                    guiyihua_min_2 = torch.min(d_p4)
                if torch.max(d_p4) >= torch.max(d_p5): #2个数中取大的
                    guiyihua_max_2 = torch.max(d_p4)
                else:
                    guiyihua_max_2 = torch.max(d_p5)

                guiyihua_max = max(guiyihua_max_1, guiyihua_max_2)
                guiyihua_min = min(guiyihua_min_1, guiyihua_min_2)

                guiyihua_scale = guiyihua_max - guiyihua_min
                ###pad
                d, h_new_p2_left, h_new_p2_right, w_new_p2_left, w_new_p2_right = Pfeature_zeropad_youxiajiao256(d, 32)
                d_p3, h_new_p3_left, h_new_p3_right, w_new_p3_left, w_new_p3_right = Pfeature_zeropad_youxiajiao128(d_p3, 16)
                d_p4, h_new_p4_left, h_new_p4_right, w_new_p4_left, w_new_p4_right = Pfeature_zeropad_youxiajiao64(d_p4, 8)
                d_p5, h_new_p5_left, h_new_p5_right, w_new_p5_left, w_new_p5_right = Pfeature_zeropad_youxiajiao32(d_p5, 4)
                d = (d - guiyihua_min) / guiyihua_scale
                d_p3 = (d_p3 - guiyihua_min) / guiyihua_scale
                d_p4 = (d_p4 - guiyihua_min) / guiyihua_scale
                d_p5 = (d_p5 - guiyihua_min) / guiyihua_scale
                d_originalsize_p2 = (d_originalsize_p2 - guiyihua_min) / guiyihua_scale
                d_originalsize_p3 = (d_originalsize_p3 - guiyihua_min) / guiyihua_scale
                d_originalsize_p4 = (d_originalsize_p4 - guiyihua_min) / guiyihua_scale
                d_originalsize_p5 = (d_originalsize_p5 - guiyihua_min) / guiyihua_scale

                conv_time = time.time() -start
                
                
                set_entropy_coder("ans")
                self.model.net_belle.update(force=True)
                start = time.time()
                out_enc = self.model.net_belle.compress(d)
                enc_time = time.time() - start
                
                for num, s in enumerate(out_enc["strings"]):
                    if num == 0:
                        with open(bin_path_y, "wb") as binary_file:
                            binary_file.write(s[0])
                    else:
                        with open(bin_path_z, "wb") as binary_file:
                            binary_file.write(s[0])
                
                size1 = os.path.getsize(bin_path_y)
                size2 = os.path.getsize(bin_path_z)
                num_pixels = self.height_temp * self.width_temp
                bpp = (size1 + size2) * 8 / num_pixels
                print(f"bpp: {bpp}, conv_time: {conv_time}, encode_time: {enc_time}")
                pbar.update()

                # self.numpixel_test5000[fname_temp].append(h_new_p2_right)
                # self.numpixel_test5000[fname_temp].append(w_new_p2_right)
                # self.numpixel_test5000[fname_temp].append(int(out_enc["shape"][0]))
                # self.numpixel_test5000[fname_temp].append(int(out_enc["shape"][1]))
                # self.numpixel_test5000[fname_temp].append(float(guiyihua_max))
                # self.numpixel_test5000[fname_temp].append(float(guiyihua_min))

        # path_save = '../new_init.json'
        # tf = open(path_save, "w")
        # json.dump(self.numpixel_test5000, tf)
        # tf.close()
        
    
    def feature_decode(self, bin_paths):
        json_ans = {}
        dump_dir = os.path.join(os.environ["OUTPUT_DIR"], f"info/{self.set_idx}.dump")
        if os.path.exists(dump_dir):
            os.remove(dump_dir)
        
        with open(os.path.join(os.environ["OUTPUT_DIR"], f"info/{self.set_idx}_coco.txt"), 'w') as of:
            of.write('ImageID,LabelName,Score,XMin,XMax,YMin,YMax\n')
            coco_classes_fname = os.path.join(os.environ["VCM_TESTDATA"], 'annotations_5k/coco_classes.txt')
            with open(coco_classes_fname, 'r') as f:
                coco_classes = f.read().splitlines()
            
            with tqdm(total=len(self.data_loader.dataset)) as pbar:
                for inputs in iter(self.data_loader):
                    images = self.model.preprocess_image(inputs)
                    fname_temp = simo_utils.simple_filename(inputs[0]["file_name"])
                    # print(self.numpixel_test5000[fname_temp])
                    self.height_temp = self.numpixel_test5000[fname_temp][0]
                    self.width_temp = self.numpixel_test5000[fname_temp][1]
                    self.numpixel_temp = self.numpixel_test5000[fname_temp][2]
                    bin_path = os.path.join(bin_paths, fname_temp) + '.bin' 
                    h_new_p2_right = self.numpixel_test5000[fname_temp][3]
                    w_new_p2_right = self.numpixel_test5000[fname_temp][4]

                    guiyihua_max = self.numpixel_test5000[fname_temp][9]
                    guiyihua_min = self.numpixel_test5000[fname_temp][10]
                    guiyihua_scale = guiyihua_max - guiyihua_min
                    set_entropy_coder("ans")
                    self.model.net_belle.update(force=True)
                    
                    bin_path_y = bin_path[:-4] + '_y' + bin_path[-4:]
                    bin_path_z = bin_path[:-4] + '_z' + bin_path[-4:]
                    read_out_enc = {}
                    read_out_enc["strings"] = []
                    read_out_enc["shape"] = torch.ones(self.numpixel_test5000[fname_temp][5], self.numpixel_test5000[fname_temp][6]).size()

                    with open(bin_path_y, 'rb') as file:
                        byte_data = list()
                        size1 = os.path.getsize(bin_path_y)
                        for i in range(size1):
                            byte_data.append(int.from_bytes(file.read(1), byteorder='big', signed=False))
                        # print(len(byte_data))
                        read_out_enc["strings"].append([bytes(byte_data)])
                        
                    
                    with open(bin_path_z, 'rb') as file:
                        size2 = os.path.getsize(bin_path_z)
                        byte_data = list()
                        for i in range(size2):
                            byte_data.append(int.from_bytes(file.read(1), byteorder='big', signed=False))
                        # print(len(byte_data))
                        read_out_enc["strings"].append([bytes(byte_data)])
                    

                    start = time.time() # test decode time
                    out_dec = self.model.net_belle.decompress(read_out_enc["strings"], read_out_enc["shape"])
                    dec_time = time.time() - start

                    start = time.time() # test conv time
                    num_pixels = self.height_temp * self.width_temp
                    # bpp = sum(len(s[0]) for s in read_out_enc["strings"]) * 8.0 / num_pixels
                    bpp = (size1 + size2) * 8 / num_pixels
                    # uint save : h_new_p2_right 0~168, w_new_p2_right 0~136, z_shape 16~22*2, float:guiyihua_min, guiyihua_max
                    # all need 12bytes, but compressai doesn't calculate z_shape bit
                    precise_bpp = (size1 + size2 + 2 + 2 + 8) * 8 / num_pixels
                    
                    

                    h_new_p3_right = int(h_new_p2_right / 2) if h_new_p2_right % 2 == 0 else None
                    h_new_p4_right = int(h_new_p2_right / 4) if h_new_p2_right % 4 == 0 else None
                    h_new_p5_right = int(h_new_p2_right / 8) if h_new_p2_right % 8 == 0 else None

                    w_new_p3_right = int(w_new_p2_right / 2) if w_new_p2_right % 2 == 0 else None
                    w_new_p4_right = int(w_new_p2_right / 4) if w_new_p2_right % 4 == 0 else None
                    w_new_p5_right = int(w_new_p2_right / 8) if w_new_p2_right % 8 == 0 else None

                    # print(h_new_p2_right, w_new_p2_right, h_new_p3_right, w_new_p3_right, h_new_p4_right, w_new_p4_right, h_new_p5_right, w_new_p5_right,)
                    
                    d_output_p3 = Pfeature_zeropad_youxiajiao128_reverse(out_dec["x_hat_p3"], 0, h_new_p3_right, 0, w_new_p3_right)
                    d_output_p2 = Pfeature_zeropad_youxiajiao128_reverse(out_dec["x_hat_p2"], 0, h_new_p2_right, 0, w_new_p2_right)
                    d_output_p4 = Pfeature_zeropad_youxiajiao128_reverse(out_dec["x_hat_p4"], 0, h_new_p4_right, 0, w_new_p4_right)
                    d_output_p5 = Pfeature_zeropad_youxiajiao128_reverse(out_dec["x_hat_p5"], 0, h_new_p5_right, 0, w_new_p5_right)

                    features_SIMO = {}
                    
                    features_SIMO["p2"] = d_output_p2 * guiyihua_scale + guiyihua_min
                    features_SIMO["p3"] = d_output_p3 * guiyihua_scale + guiyihua_min
                    features_SIMO["p4"] = d_output_p4 * guiyihua_scale + guiyihua_min
                    features_SIMO["p5"] = d_output_p5 * guiyihua_scale + guiyihua_min
                    features_SIMO["p6"] = F.max_pool2d(features_SIMO["p5"], kernel_size=1, stride=2, padding=0)

                    self.dump.write_layers(dump_dir, features_SIMO["p2"])

                    outputs = self.forward_front(inputs, images, features_SIMO)  # images是float64
                    conv_time = time.time() - start
                    frame_ans =  {
                        "bpp": bpp,
                        "decoding_time": dec_time,
                        "precise_bpp": precise_bpp,
                        "conv_part2_time": conv_time,
                    }
                    json_ans[fname_temp] = frame_ans
                    print(fname_temp, frame_ans)
                    self.evaluator.process(inputs, outputs)

                    outputs = outputs[0]
                    imageId = os.path.basename(fname_temp)
                    classes = outputs['instances'].pred_classes.to('cpu').numpy()
                    scores = outputs['instances'].scores.to('cpu').numpy()
                    bboxes = outputs['instances'].pred_boxes.tensor.to('cpu').numpy()
                    H, W = outputs['instances'].image_size
                    bboxes = bboxes / [W, H, W, H]
                    bboxes = bboxes[:, [0, 2, 1, 3]]
                    for ii in range(len(classes)):
                        coco_cnt_id = classes[ii]
                        class_name = coco_classes[coco_cnt_id]
                        rslt = [imageId, class_name, scores[ii]] + \
                            bboxes[ii].tolist()
                        o_line = ','.join(map(str, rslt))
                        of.write(o_line + '\n')
                    pbar.update()

        tf = open(self.path_bppsave, "w")
        json.dump(json_ans, tf)
        tf.close()
        bpp_sum = 0
        actual_bpp_sum = 0
        decode_time_sum = 0
        i_count = 0
        for key in json_ans:
            frame_ans = json_ans[key]
            bpp_sum = bpp_sum + frame_ans["bpp"]
            actual_bpp_sum = actual_bpp_sum + frame_ans["precise_bpp"]
            decode_time_sum = decode_time_sum + frame_ans["decoding_time"]
            i_count = i_count + 1
            # print('i_count: %d, bpp: %8.4f, %s' % (i_count, bpp_test5000[key][0], key))
        print('average bpp: %9.6f' % (bpp_sum / i_count))
        print('average precise bpp: %9.6f' % (actual_bpp_sum / i_count))
        print('average decoding time: %9.6f' % (decode_time_sum / i_count))

        self.det_eval.forward()
       


class DetectEval(Eval):
    def prepare_part(self, myarg, data_name="pick"):
        print("Loading", data_name, "...")
        simo_utils.pick_coco_exp(data_name, myarg)
        self.data_loader = build_detection_test_loader(self.cfg, data_name)
        self.evaluator = COCOEvaluator(data_name, self.cfg, False)
        self.evaluator.reset()
        print(data_name, "Loaded")

