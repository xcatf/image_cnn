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

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    width_shift_range=0.4,
    height_shift_range=0.4,
    rotation_range=90,
    zoom_range=0.7,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=keras.applications.xception.preprocess_input)

test_datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=keras.applications.xception.preprocess_input)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    save_to_dir=image_train_save_dir,
    target_size=img_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    dataset_dir,
    save_to_dir=image_train_save_dir,
    target_size=img_size,
    class_mode='categorical')

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=13)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=7, mode='auto', factor=0.2)

tensorboard = keras.callbacks.TensorBoard(log_dir=train_log_dir)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples//train_generator.batch_size,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples//validation_generator.batch_size,
    callbacks=[early_stop, reduce_lr, tensorboard])

model.save(model_dir + 'medecine_model_v1.h5')
with open(model_info_dir + 'model_v1_summary.txt','w') as f:
    with redirect_stdout(f):
        model.summary()

# accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(model_info_dir + 'model_v1_acc.jpg')

# loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(model_info_dir + 'model_v1_loss.jpg')

