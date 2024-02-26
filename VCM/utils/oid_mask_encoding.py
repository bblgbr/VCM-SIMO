# Copyright (c) 2020-2021 Nokia Corporation and/or its subsidiary(-ies). All rights reserved.
# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

#
# follows instructions from https://gist.github.com/pculliton/209398a2a52867580c6103e25e55d93c

import base64
import numpy as np
from pycocotools import _mask as coco_mask
import zlib

def encode_binary_mask(mask):
 # check mask data type and shape
#  assert mask.dtype==np.bool, print('mask must be a binary mask')

 mask = np.squeeze(mask)
 assert len(mask.shape)==2, print('mask must be a 2D tensro') 

 # prepare data
 mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
 mask_to_encode = mask_to_encode.astype(np.uint8)
 mask_to_encode = np.asfortranarray(mask_to_encode)

 # encode
 encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

 # compress
 binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
 base64_str = base64.b64encode(binary_str)
 return base64_str
