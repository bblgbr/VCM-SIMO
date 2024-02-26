set -e
ulimit -c 0
detection_openimage.py -i 5 -b ${OUTPUT_DIR}/info/point5.json -m feature_decode -p 5 --bin-paths ${BITSTERM_DIR}/5