import os 
import tensorflow as tf
from tensorflow.keras.application.vgg16 import preprocess_input

form member1_dataset import IMAGE_SIZE,get_datasets
from member2_model import my_loss
form member3_train import MODEL_FILE,CLASS_NAMES_FILE

TEST_IMAGE = ""

def load_class_name():
file = open(CLASS_NAME_FILE,"r")
lines = file.readlines()
file.close()

class_name = []
for line in lines:
name = line.strip()
if name!= "":
class_name.append(name)

return class_names

def prepare_single_image(image_path):

image = tf.io.read_file(image_path)
image = tf.image.decode_image(image, channels=3, expand_antimations=False)
image.set_shape([None, None, 3])
image = tf.image.resize(image,[IMAGE_SIZE, IMAGE_SIZE])
image = preprocess_input(image)
image = tf.expand_dims(image,0)

  return image

def predict_one_image(model,image_path,class_names):

image = prepare_single_image(image_path)
prediction = model.predict(image)

class_index = int(tf.argmax(predication[0]))
fruit_name = class_names[class_index]
confidence = round(float(prediction[0][class_index])*100,2)

print("Image file:",image_path)
print("predicted fruit:",fruit_name)
print("Confidence:", confidence,"%")

def run():
 if not os.path.exists(MODEL_FILE)
  print("Model file not found:", MODEL_FILE)
  print("Please run member3_train.py first to train and save the model")
   return

Print("Loading saved model...")
model = tf.keras.models.load_model(MODEL_FILE, Custom_objects ={"my_loss": my_loss}
)
class_name = load_class_names()

print("Evaluating model on validation data ...")
_, val_data,_ =get_datasets()
model.evaluate(val_data)

if TEST_IMAGE != "":
print("")
print("Running predication on single image...")
if os.path.exists(TEST_IMAGE):
      predict_one_image(model, TEST_IMAGE, class_names)
      else:
      print("Image not found:", TEST_IMAGE)
      print("please check the TEST_IMAGE path at the top of this file")

  run()
