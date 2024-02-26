set -e
ulimit -c 0
detection_openimage.py -i 3 -b ${OUTPUT_DIR}/info/point3.json -m feature_decode -p 3 --bin-paths ${BITSTERM_DIR}/3