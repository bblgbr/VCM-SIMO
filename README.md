# SIMO Openimage Detection and Segmentation Anchor

`/* The copyright in this software is being made available under the BSD * License, included below. This software may be subject to other third party * and contributor rights, including patent rights, and no such rights are * granted under this license. ** Copyright (c) 2023, ISO/IEC * All rights reserved. ** Redistribution and use in source and binary forms, with or without * modification, are permitted provided that the following conditions are met: ** * Redistributions of source code must retain the above copyright notice, * this list of conditions and the following disclaimer. * * Redistributions in binary form must reproduce the above copyright notice, * this list of conditions and the following disclaimer in the documentation * and/or other materials provided with the distribution. * * Neither the name of the ISO/IEC nor the names of its contributors may * be used to endorse or promote products derived from this software without * specific prior written permission. ** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF * THE POSSIBILITY OF SUCH DAMAGE. */E`

This work has been accepted in ICIP2024

## Archive

- CompressAI: compressai package
- detectron2: detectron2 package
- tf-models: tensorflow model package to help detection
- VCM: our SIMO method

>If you want to encode again you need to remove the stream_SIMO/openimg_segmen_plyr_anch2_enc_nosave/*/*.bin because in the code if find the *.bin it will skip encoding
>If you only crosscheck, please go to decode directly

## Set up environment

### Conda create a environment

```bash
conda create -n SIMO-DetSeg-Openimage python=3.8
conda activate SIMO-DetSeg-Openimage
```

> Note we use cuda11.7 version, you need to choose your version by yourselves
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

### Detectron2 setup

We provide the detectron2, so you don't need to git and could install it directly.

```bash
python -m pip install -e detectron2
```

> Note that if you meet the error :NotADirectoryError: [Errno 20] Not a directory: 'hipconfig', you could
vim ~/anaconda3/envs/SIMO-Segmentation-Openimage/lib/python3.8/site-packages/torch/utils/hipify/cuda_to_hip_mappings.py and change the line 37 to add NotADirectoryError
`except (FileNotFoundError, PermissionError, NotADirectoryError):`

### Tensorflow object detection setup

We provide the detectron2, so you don't need to git and could install it directly.

```bash
cd tf-models/research/
conda install -yc anaconda protobuf
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```

>If tf-models is correctly installed, by running the following line, you should see something like this:
>Ran 24 tests in 22.795s

```bash
python object_detection/builders/model_builder_tf2_test.py
```

### Compressai setup

```bash
cd ../../CompressAI
pip install -U pip && pip install -e .
cd ..
```

### requirement setup

```bash
chmod -R +x ./VCM
cd VCM
```

### weight and dataset setup

You need to download the weight directory and dataset from the following link and move to `./VCM`

`https://bhpan.buaa.edu.cn/link/AA7D2A55E9E55D446AB637B236D61A2847`

- CompressAI
- detectron2
- tf-models
- VCM
  - dataset
  - weights
  - stream_SIMO

## SIMO step

>**If you want to check the whole pipline, you could do the following steps. Otherwise, you could go to decode step directly and finish the crosscheck !**

Since we ues the learning method, encoding and decoding will use gpu to speed up. If you want to use cpu, you can modify the `env_det.sh` and `env_seg.sh` to change the `DEVICE` variable to 'cpu'. Also, you could compare the running time between `cpu` and `gpu`.

### Encode Detection task

```bash
source env_det.sh
cd scripts/det_openimage
./seqrun.sh run_openimage_detection_chenganchor_enc_SIMO.sh
```

### Decode Detection task

```bash
source env_det.sh
cd scripts/det_openimage
./seqrun.sh run_openimage_detection_chenganchor_dec_SIMO.sh
cd ..
```

You could go to `./VCM/sim_SIMO/openimg_objdetc_postplyr_anch2_dec/{idx}` to check the output from stderr and stdout file to see if decoding successfully.

Example:

```json
32bd80d38fc5b085 {'bpp': 0.01156045751633987, 'decoding_time': 6.820584774017334, 'precise_bpp': 0.011683006535947713, 'conv_part2_time': 0.9632875919342041}
```

>Note:Since the method is based on the autoregressive method of learning, the decoding time is quite long, about 12 hours to run one point, so you can run multiple scripts on different gpus. If you do so, you need to change the 'CUDA_VISIBLE_DEVICES' in `env_det.sh` and source in different bash. I show an example with 2 scripts, you could write 6 scripts.

```bash
Example:
source env_det.sh
cd scripts/seg_openimage
./seqrun.sh run_openimage_detection_chenganchor_dec_SIMO_1.sh
source env_det.sh
./seqrun.sh run_openimage_detection_chenganchor_dec_SIMO_2.sh
```

### Encode Segmentation task

>**If you want to check the whole pipline, you could do the following steps. Otherwise, you could go to decode step directly and finish the crosscheck !**

```bash
source env_seg.sh
cd scripts/seg_openimage
./seqrun.sh run_openimage_detection_chenganchor_enc_SIMO.sh
```

### Decode Segmentation task

```bash
source env_det.sh
cd scripts/seg_openimage
./seqrun.sh run_openimage_segmentation_chenganchor_dec_SIMO.sh
```

You could go to `./VCM/sim_SIMO/openimg_segmen_postplyr_anch2_dec/{idx}` to check the output from stderr and stdout file to see if decoding successfully.

>Note:Since the method is based on the autoregressive method of learning, the decoding time is quite long, about 8 hours to run one point, so you can run multiple scripts on different gpus. If you do so, you need to change the 'CUDA_VISIBLE_DEVICES' in `env_seg.sh` and source in different bash. I show an example with 2 scripts, you could write 6 scripts.

```bash
source env_det.sh
cd scripts/seg_openimage
./seqrun.sh run_openimage_segmentation_chenganchor_dec_SIMO_1.sh
source env_det.sh
./seqrun.sh run_openimage_segmentation_chenganchor_dec_SIMO_2.sh
```

## Get the result

Do the following command and get the csv file of all crosscheck result from `./VCM/result_crosscheck/{task}/output`

```bash
get_result.py -m det
get_result.py -m seg
```

### Get all the dump

You need run this script in detection shell because it reply on the environment variable.

```bash
get_all_dump_file.py --src ${OUTPUT_DIR}/info --dest ./detection_dump_crosscheck
```

You need run this script in segmentation shell because it reply on the environment variable.

```bash
get_all_dump_file.py --src ${OUTPUT_DIR}/info --dest ./segmentation_dump_crosscheck
```

### Compare dump

```bash
compare.py --src1 detection_dump --src2 detection_dump_crosscheck
compare.py --src1 segmentation_dump --src2 segmentation_dump_crosscheck
```

If check successfully, it will print `All OK!`.
