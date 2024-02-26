#
# Portable environment for distributable proposal packages
#

# All datasets under here
export VCM_ROOT=`pwd`

export VCM_TESTDATA=`pwd`/dataset/seg_openimage
export DETECTRON2_DATASETS=`pwd`/dataset/seg_openimage/validation

# Set directory for weights (for proposal(s) using learned modules)
export WEIGHTS_DIR=`pwd`/weights

export BITSTERM_DIR=`pwd`/stream_SIMO/openimg_segmen_plyr_anch2_enc

export OUTPUT_DIR=`pwd`/result_crosscheck/seg_openimage

export SETTING_DIR=`pwd`/settings/seg_openimage

export PATH=`pwd`/bin:${PATH}
export PYTHONPATH=`pwd`:${PYTHONPATH}

export DETECTRON2=`pwd`/../detectron2

# For JDE
# export JDE=`pwd`/Towards-Realtime-MOT

# Constrain DNNL to avoid AVX512, which leads to non-deterministic operation across different CPUs..
export DNNL_MAX_CPU_ISA=AVX2

# For proposals and crosschecks, use 'cpu' (with AVX512 disabled)
# For internal testing, OK to use 'cuda' for faster execution
export DEVICE=cuda
export CUDA_VISIBLE_DEVICES='2'
