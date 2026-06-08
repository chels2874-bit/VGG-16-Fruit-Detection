import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

def my_loss(y_true, y_pred):
  return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
  
def build_vgg16_model(num_classes):
  
  vgg_base = VGG16(
  weights= "imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
  )

for layer in vgg_base.layers:
  layer.trainable = False

model = Sequential()
model.add(vgg_base)
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation="softmax")

model.summary()

return model
