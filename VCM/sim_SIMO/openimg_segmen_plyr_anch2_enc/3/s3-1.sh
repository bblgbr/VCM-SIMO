set -e
ulimit -c 0
segmentation_openimage.py -i 3 -m feature_encode -p 3 --bin-paths ${BITSTERM_DIR}/3