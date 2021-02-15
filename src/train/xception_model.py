# 利用Keras构建Xception模型
import keras, os
from contextlib import redirect_stdout
img_size = (299, 299, 3)
 # 预训练Xception权重
base_weight_dir = '../../resource/model/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
model_into_dir = '../../resource/model_info/'
base_model = keras.applications.xception.Xception(include_top=False,
                                                 weights=base_weight_dir,
                                                 input_shape=img_size,
                                                 pooling='avg')
# 最后全连接层对628种中药材分类
model = keras.layers.Dense(628, activation='softmax', name='predictions')(base_model.output)
model = keras.Model(base_model.input, model)
for layer in base_model.layers:
    layer.trainable = False