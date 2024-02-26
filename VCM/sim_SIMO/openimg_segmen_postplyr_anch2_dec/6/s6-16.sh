set -e
ulimit -c 0
segmentation_openimage.py -i 6 -b ${OUTPUT_DIR}/info/point6.json -m feature_decode -p 6 --bin-paths ${BITSTERM_DIR}/6