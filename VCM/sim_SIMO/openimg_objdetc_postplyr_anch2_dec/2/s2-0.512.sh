set -e
ulimit -c 0
detection_openimage.py -i 2 -b ${OUTPUT_DIR}/info/point2.json -m feature_decode -p 2 --bin-paths ${BITSTERM_DIR}/2