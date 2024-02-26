set -e
ulimit -c 0
segmentation_openimage.py -i 6 -m feature_encode -p 6 --bin-paths ${BITSTERM_DIR}/6