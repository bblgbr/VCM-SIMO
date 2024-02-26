# @Time : 2023/7/26 13:36
# @Developer : zzf
# @FileName: write_sim.py
# @Software: PyCharm
import os
point = range(1,7)
lambd = {1: 0.256, 2:0.35, 3:1, 4:2, 5:4, 6:16}

for i in point: # i 为QP标号
    if not os.path.exists(f"./openimg_segmen_postplyr_anch2_dec/{i}"):
        os.makedirs(f"./openimg_segmen_postplyr_anch2_dec/{i}")
    
    if os.path.exists(f"./openimg_segmen_postplyr_anch2_dec/{i}/s{i}-{lambd[i]}.sh"):
        os.remove(f"./openimg_segmen_postplyr_anch2_dec/{i}/s{i}-{lambd[i]}.sh")
    with open(f"./openimg_segmen_postplyr_anch2_dec/{i}/s{i}-{lambd[i]}.sh", "a") as f:
        f.write("set -e\n")
        f.write("ulimit -c 0\n")
        f.write("segmentation_openimage.py -i %d -b ${OUTPUT_DIR}/info/point%d.json -m feature_decode -p %d --bin-paths ${BITSTERM_DIR}/%d" % (i, i, i, i))
        f.flush()


