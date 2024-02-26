import numpy as np
import cv2
import torch
import json
def save_feature_map(filename, features, debug=False):  # 输入y
    
    _min = -0.09198
    _max = 0.08042
    _scale = 1000 / (_max - _min)
    features_guiyi = (features - _min) * _scale
    feat = [features_guiyi.squeeze()]
    width_list = [16]
    height_list = [12]
    tile_big = np.empty((0, feat[0].shape[2] * width_list[0]))
    for blk, width, height in zip(feat, width_list, height_list):
        big_blk = np.empty((0, blk.shape[2] * width))
        for row in range(height):
            big_blk_col = np.empty((blk.shape[1], 0))
            for col in range(width):
                tile = blk[col + row * width].cpu().numpy()
                if debug:
                    cv2.putText(tile, f"{col + row * width}", (32, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                                1, )
                big_blk_col = np.hstack((big_blk_col, tile))
            big_blk = np.vstack((big_blk, big_blk_col))
        tile_big = np.vstack((tile_big, big_blk))
    tile_big = tile_big.astype(np.uint16)
    cv2.imwrite(filename, tile_big)


def feat2feat(fname):  # 将读入编码后特征图片转为相应通道数的tensor
    pyramid = {}
    _min = -0.09198
    _max = 0.08042
    _scale = 1000 / (_max - _min)
    # _min = -0.23
    # _max = 0.16
    # _scale = 511 / (_max - _min)
    png = cv2.imread(fname, -1).astype(np.float32)
    # print(png.shape[0] // 12, png.shape[1] // 16)
    pyramid["y"] = feature_slice(png, [png.shape[0] // 12, png.shape[1] // 16])
    pyramid["y"] = torch.unsqueeze(pyramid["y"], 0)
    pyramid["y"] = pyramid["y"] / _scale + _min
    # 加了下面这几句弄到cuda
    import os
    pyramid["y"] = pyramid["y"].to(os.environ["DEVICE"])
    return pyramid


def feature_slice(image, shape):
    height = image.shape[0]
    width = image.shape[1]

    blk_height = shape[0]
    blk_width = shape[1]
    blk = []

    for y in range(height // blk_height):
        for x in range(width // blk_width):
            y_lower = y * blk_height
            y_upper = (y + 1) * blk_height
            x_lower = x * blk_width
            x_upper = (x + 1) * blk_width
            blk.append(image[y_lower:y_upper, x_lower:x_upper])
    feature = torch.from_numpy(np.array(blk))
    return feature