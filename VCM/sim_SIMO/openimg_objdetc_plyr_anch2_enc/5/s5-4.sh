set -e
ulimit -c 0
detection_openimage.py -i 5 -m feature_encode -p 5 --bin-paths ${BITSTERM_DIR}/5