import os

point = range(1,7)
lambd = {1: '0.256', 2:'0.512', 3:1, 4:'2', 5:'4', 6:'16'}

if os.path.exists("./run_openimage_segmentation_chenganchor_enc_nosave_SIMO.sh"):
    os.remove("./run_openimage_segmentation_chenganchor_enc_nosave_SIMO.sh")
with open("./run_openimage_segmentation_chenganchor_enc_nosave_SIMO.sh", "a") as f:
    for i in point:
        f.write("./sim_SIMO/openimg_segmen_plyr_anch2_enc_nosave/%d/s%d-%s.sh\n" % (i, i, lambd[i]))
        f.flush()

if os.path.exists("./run_openimage_segmentation_chenganchor_dec_nosave_SIMO.sh"):
    os.remove("./run_openimage_segmentation_chenganchor_dec_nosave_SIMO.sh")
with open("./run_openimage_segmentation_chenganchor_dec_nosave_SIMO.sh", "a") as f:
    for i in point:
        f.write("./sim_SIMO/openimg_segmen_postplyr_anch2_dec_nosave/%d/s%d-%s.sh\n" % (i, i, lambd[i]))
        f.flush()


