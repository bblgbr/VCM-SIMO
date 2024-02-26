# @Time : 2023/7/26 13:36
# @Developer : zzf
# @FileName: write_sim.py
# @Software: PyCharm
import os
point = range(1,7)
lambd = {1: 0.256, 2:0.35, 3:1, 4:2, 5:4, 6:16}

for i in point: # i 为QP标号
    if not os.path.exists(f"./openimg_segmen_plyr_anch2_enc/{i}"):
        os.makedirs(f"./openimg_segmen_plyr_anch2_enc/{i}")
    
    if os.path.exists(f"./openimg_segmen_plyr_anch2_enc/{i}/s{i}-{lambd[i]}.sh"):
        os.remove(f"./openimg_segmen_plyr_anch2_enc/{i}/s{i}-{lambd[i]}.sh")
    with open(f"./openimg_segmen_plyr_anch2_enc/{i}/s{i}-{lambd[i]}.sh", "a") as f:
        f.write("set -e\n")
        f.write("ulimit -c 0\n")
        f.write("segmentation_openimage.py -i %d -m feature_encode -p %d --bin-paths ${BITSTERM_DIR}/%d" % (i, i, i))
        f.flush()
    

