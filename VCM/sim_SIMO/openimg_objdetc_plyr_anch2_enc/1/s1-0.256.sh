set -e
ulimit -c 0
detection_openimage.py -i 1 -m feature_encode -p 1 --bin-paths ${BITSTERM_DIR}/1