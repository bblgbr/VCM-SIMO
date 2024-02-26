import torch
# check torch、cuda、cudnn version
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())

# whether could use gpu
flag = torch.cuda.is_available()
print(flag)

# check the detectron2 package's location
import detectron2
print(detectron2)

# check the compressai package's location
import compressai
print(compressai)

# check the python path
import os
print(os.environ["PYTHONPATH"])
