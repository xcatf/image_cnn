import keras
from xception_model import model
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
# 训练xception模型
img_size = (299, 299)
dataset_dir = '../../resource/dataset'
train_log_dir = '../../resource/train_log'
image_train_save_dir = '../../resource/train_image'
model_dir = '../../resource/model/'
model_info_dir = '../../resource/model_info/'
label_dir = '../../resource/medicine_label/medicine_label.csv'
for layer in model.layers:
    layer.trainable = False

for layer in model.layers[126:132]:
    layer.trainable = True

for layer in model.layers:
    if(layer.trainable == True):
        print(layer)