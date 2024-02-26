import argparse
import subprocess
import os
import pandas as pd
import csv
import json

class EVAL_mAP:
    def __init__(self, index, mode) -> None:
        self.set_idx = index
        self.mode = mode
        if mode == 'det':
            self.lambdalist = {1: 0.256, 2:0.512, 3:1, 4:2, 5:4, 6:16}
        elif mode == 'seg':
            self.lambdalist = {1: 0.256, 2:0.35, 3:1, 4:2, 5:4, 6:16}

    def forward(self):
        """
           finish the test pipeline
        """
        self.conversion()
        self.get_mAP()

    def get_mAP(self):
        """
           get the mAP reuslt and write to {self.set_idx}_AP.txt
        """
        if self.mode == 'seg':
            cmd = f"python {os.path.join(os.environ['VCM_ROOT'], 'utils')}/oid_challenge_evaluation.py" \
            f" --input_annotations_boxes  {os.path.join(os.environ['VCM_TESTDATA'], 'annotations_5k/segmentation_validation_bbox_5k.csv')}" \
            f" --input_annotations_labels {os.path.join(os.environ['VCM_TESTDATA'], 'annotations_5k/segmentation_validation_labels_5k.csv')}" \
            f" --input_class_labelmap     {os.path.join(os.environ['VCM_TESTDATA'], 'annotations_5k/coco_label_map.pbtxt')}" \
            f" --input_annotations_segm   {os.path.join(os.environ['VCM_TESTDATA'], 'annotations_5k/segmentation_validation_masks_5k.csv')}" \
            f" --input_predictions        {os.path.join(os.environ['OUTPUT_DIR'], f'info/{self.set_idx}_oi.txt')}" \
            f" --output_metrics           {os.path.join(os.environ['OUTPUT_DIR'], f'info/{self.set_idx}_AP.txt')}"
            print(">>>> cmd: ", cmd)
            subprocess.call([cmd], shell=True)
        elif self.mode == 'det':
            cmd = f"python {os.path.join(os.environ['VCM_ROOT'], 'utils')}/oid_challenge_evaluation.py" \
            f" --input_annotations_boxes  {os.path.join(os.environ['VCM_TESTDATA'], 'annotations_5k/detection_validation_5k_bbox.csv')}" \
            f" --input_annotations_labels {os.path.join(os.environ['VCM_TESTDATA'], 'annotations_5k/detection_validation_labels_5k.csv')}" \
            f" --input_class_labelmap     {os.path.join(os.environ['VCM_TESTDATA'], 'annotations_5k/coco_label_map.pbtxt')}" \
            f" --input_predictions        {os.path.join(os.environ['OUTPUT_DIR'], f'info/{self.set_idx}_oi.txt')}" \
            f" --output_metrics           {os.path.join(os.environ['OUTPUT_DIR'], f'info/{self.set_idx}_AP.txt')}"
            print(">>>> cmd: ", cmd)
            subprocess.call([cmd], shell=True)
            
    def conversion(self):
        """
           convert the coco format to ori format
        """
        index = self.set_idx
        coco_fname = os.path.join(os.environ["OUTPUT_DIR"], f"info/{index}_coco.txt")
        oid_fname = os.path.join(os.environ["OUTPUT_DIR"], f"info/{index}_oi.txt")
        selected_coco_classes_fname = os.path.join(os.environ["VCM_TESTDATA"], "annotations_5k/selected_classes.txt")
        # unify class name by replacing space to underscore
        def unify_name(class_name):
            return class_name.replace(' ', '_')
        # selected coco classes
        with open(selected_coco_classes_fname, 'r') as f:
            selected_classes = [unify_name(x) for x in f.read().splitlines()]
        # load coco output data file
        coco_output_data = pd.read_csv(coco_fname)
        # generate output
        of = open(oid_fname, 'w')
        #write header
        #of.write('ImageID,LabelName,Score,XMin,XMax,YMin,YMax\n')
        of.write(','.join(coco_output_data.columns)+'\n')
        # iterate all input files
        for idx, row in coco_output_data.iterrows():
            fields = row.tolist()
            #coco_id = fields[1]
            coco_id = unify_name(row['LabelName'])
            if coco_id in selected_classes:
                oid_id = coco_id
                row['LabelName'] = oid_id
                o_line = ','.join(map(str,row))
                of.write(o_line + '\n')
        of.close()
    
    def write_result(self):
        """
            get all six points average result
        """
        csv_path = os.path.join(os.environ["OUTPUT_DIR"], f"output/SIMO_{self.mode}_result.csv")
        header = ['point', 'precise_bpp', "mAP", 'decoding_time', 'conv_part2_time']
        datas = []

        for i in range(1, 3):
            SIMO_ans = {}
            precise_bpp = 0
            decoding_time = 0
            conv_part2_time = 0
            mAP = 0
            # get mAP result
            mAP_path = os.path.join(os.environ["OUTPUT_DIR"], f"info/{i}_AP.txt")
            with open(mAP_path, "rt") as ap_f:
                mAP = ap_f.readline()
                mAP = mAP.split(",")[1][:-1]

            # get decode information including bpp, dec time, inverse conv time
            json_path = os.path.join(os.environ["OUTPUT_DIR"], f"info/point{i}.json")
            with open(json_path, 'r') as file:
                dec_info = json.load(file)
                for img_name, info in dec_info.items():
                    precise_bpp += info["precise_bpp"]
                    decoding_time += info["decoding_time"]
                    conv_part2_time += info["conv_part2_time"]
            
            # save all average result to a list
            num = len(dec_info)
            SIMO_ans["mAP"] = float(mAP) * 100
            SIMO_ans["point"] = f"{self.mode}_{i}_lambda{self.lambdalist[i]}"
            SIMO_ans["precise_bpp"] = precise_bpp / num
            SIMO_ans["decoding_time"] = decoding_time / num
            SIMO_ans["conv_part2_time"] = conv_part2_time / num
            datas.append(SIMO_ans)
        
        # write data to xls
        with open(csv_path, 'w', newline='',encoding='utf-8') as f: 
            writer = csv.DictWriter(f,fieldnames=header) # 提前预览列名，当下面代码写入数据时，会将其一一对应。
            writer.writeheader()  # 写入列名
            writer.writerows(datas) # 写入数据
        print(f"write to {csv_path}")