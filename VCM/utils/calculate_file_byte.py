import json

# 指定 JSON 文件的路径
json_file_path = "./dataset/annotations_5k/Openimage_numpixel_test5000_new.json"

# 打开 JSON 文件并加载数据
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# 计算 JSON 字典的长度
print(len(data))
max_1 = - 1000
max_2 = - 1000
max_3 = - 1000
max_4 = - 1000
min_1 = 1000
min_2 = 1000
min_3 = 1000
min_4 = 1000
for i in data:
    max_1 = data[i][3] if data[i][3] > max_1 else max_1
    max_2 = data[i][4] if data[i][4] > max_2 else max_2
    max_3 = data[i][5] if data[i][5] > max_3 else max_3
    max_4 = data[i][6] if data[i][6] > max_4 else max_4
    min_1 = data[i][3] if data[i][3] < min_1 else min_1
    min_2 = data[i][4] if data[i][4] < min_2 else min_2
    min_3 = data[i][5] if data[i][5] < min_3 else min_3
    min_4 = data[i][6] if data[i][6] < min_4 else min_4

print(max_1, max_2, max_3, max_4)
print(min_1, min_2, min_3, min_4)
    