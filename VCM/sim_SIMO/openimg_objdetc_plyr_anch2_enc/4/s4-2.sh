set -e
ulimit -c 0
detection_openimage.py -i 4 -m feature_encode -p 4 --bin-paths ${BITSTERM_DIR}/4