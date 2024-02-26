set -e
ulimit -c 0
segmentation_openimage.py -i 2 -m feature_encode -p 2 --bin-paths ${BITSTERM_DIR}/2