import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequentials 
from tensorflow.keras.layers import Dense, Flatten, Dropout

def my_loss(y_true, y_pred):

  vgg_base = VGG16(
  weights= "imagenet",
    includde_top=False,
    input_shap= (224, 224, 3)
  )

for ayer in vgg_base.layers:
  layer.trainable = False

model = Sequential()
model.add(vgg_base)
model.add(Flatten())
model.add(Dense(128, activations="relu"))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation="softmax")

model.summary()

return model
