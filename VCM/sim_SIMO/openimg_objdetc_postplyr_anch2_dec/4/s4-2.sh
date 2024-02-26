set -e
ulimit -c 0
detection_openimage.py -i 4 -b ${OUTPUT_DIR}/info/point4.json -m feature_decode -p 4 --bin-paths ${BITSTERM_DIR}/4