set -e
ulimit -c 0
segmentation_openimage.py -i 5 -m feature_encode -p 5 --bin-paths ${BITSTERM_DIR}/5