import os 
import random 
import tensorflow as tf
from tensorflow.keras.application.vgg16 import preprocess_input
IMAGE_SIZE = 224 
BATCH_SIZE =16 
NUM_CLASSES= 15
def get_fruit_names():

  fruit_names =[]
  for  item in os.listdir(DATASET_PATH):
      full_path= os.path.join(DATASET_PATH , item )
      if os.path.isdir(full_path):
          furit_name.append(item)
  fruit_name.sort()
  print("Total class found", len(fruit_names))
  print("class name ", fruit_names)

  return fruit_names

def collect_all_images(fruit_names):
   

    all_images = []

    for i in range(len(fruit_names)):
        fruit_name = fruit_names[i]
        fruit_folder = os.path.join(DATASET_PATH, fruit_name)

        for root_dir, sub_dirs, files in os.walk(fruit_folder):
            for file_name in files:
                is_jpg = file_name.lower().endswith(".jpg")
                is_jpeg = file_name.lower().endswith(".jpeg")
                is_png = file_name.lower().endswith(".png")

                if is_jpg or is_jpeg or is_png:
                    image_path = os.path.join(root_dir, file_name)
                    all_images.append([image_path, i])

    
    random.seed(42)
    random.shuffle(all_images)

    return all_images


def load_and_process_image(image_path, label):
  

    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = preprocess_input(image)       
    label = tf.one_hot(label, NUM_CLASSES)

    return image, label


def create_tf_dataset(image_list):
   

    paths = []
    labels = []

    for item in image_list:
        paths.append(item[0])
        labels.append(item[1])

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(load_and_process_image)
    dataset = dataset.batch(BATCH_SIZE)

    return dataset


def get_datasets():
    
    fruit_names = get_fruit_names()
    all_images = collect_all_images(fruit_names)

    
    split = int(len(all_images) * 0.8)
    train_images = all_images[:split]
    val_images = all_images[split:]

    train_dataset = create_tf_dataset(train_images)
    val_dataset = create_tf_dataset(val_images)

    print("Total images:", len(all_images))
    print("Training images:", len(train_images))
    print("Validation images:", len(val_images))

    return train_dataset, val_dataset, fruit_names







