set -e
ulimit -c 0
segmentation_openimage.py -i 1 -b ${OUTPUT_DIR}/info/point1.json -m feature_decode -p 1 --bin-paths ${BITSTERM_DIR}/1