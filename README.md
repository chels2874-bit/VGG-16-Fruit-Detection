# Fruit Recognition using VGG16

## About the Project

This is our mini project for the Deep Learning course. We had to classify fruit images into 15 different categories using a CNN model. We chose **Project 3 — Fruit Recognition** from the given list.

We used **Transfer Learning** with **VGG16** because training a deep CNN from scratch would take too much time and data. VGG16 is already trained on ImageNet which has over 1 million images, so it already knows how to detect edges, textures and shapes. We just added our own layers on top of it to classify 15 fruit types. This saved us a lot of training time and gave better accuracy compared to building a model from scratch.

---

## Team Members

| Member | File | What they did | 
|--------|------|----------------|
|JATIN KUMAR JATOLIA| `member1_dataset.py` | Wrote the code to read the dataset folder, collect all image paths, assign labels, shuffle and split into 80% train and 20% validation, and create the TensorFlow data pipeline |
| CHELS BANSHIWAL | `member2_model.py` | Built the VGG16 model with our custom classification layers on top and wrote the custom loss function |
| G.VIKAS JOEL | `member3_train.py` | Wrote the training code — compiled the model with Adam optimizer and our loss function, ran training for 10 epochs and saved the model |
| SHRIKANT METERE | `member4_predict.py` | Wrote code to load the saved model, evaluate it on validation data and predict fruit from a single image with confidence score | 

---

## Dataset

- **Name:** Fruit Recognition
- **Download:** https://www.kaggle.com/datasets/chrisfilo/fruit-recognition
- **Size:** ~8 GB
- **Classes:** 15 fruit types
- **Format:** JPG / PNG images

The dataset has 15 subfolders, each named after a fruit. All images are inside those folders. We read the folder names as class labels so no separate label file is needed.

---

## Model — VGG16

We used **VGG16** because our professor specified it and also because it is one of the most reliable CNN architectures for image classification. It was developed by Oxford's Visual Geometry Group and it performs very well even on custom datasets through transfer learning.

We **froze all the VGG16 layers** because we didn't want to disturb the weights it already learned from ImageNet. We only trained the layers we added on top — a Dense layer with 128 neurons, a Dropout layer and a final output layer with 15 neurons (one per fruit class).

```
VGG16 Base (frozen, pretrained on ImageNet)
     ↓
Flatten
     ↓
Dense — 128 neurons, ReLU activation
     ↓
Dropout — 0.3 (to reduce overfitting)
     ↓
Dense — 15 neurons, Softmax (final output)
```

---

## Loss Function

We used **Categorical Crossentropy** as our loss function because this is a multi-class classification problem with 15 classes. Categorical crossentropy works well here because it compares the predicted probability distribution with the actual one-hot encoded label and penalizes the model more when it is confidently wrong.

As required in the project, we wrapped it inside our own custom function:

```python
def my_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
```

---

## Optimizer

We used the **Adam optimizer** with a learning rate of **0.0001** because Adam adapts the learning rate automatically for each parameter which makes training faster and more stable compared to plain SGD. We kept the learning rate small (0.0001) because we are doing transfer learning — a high learning rate would overwrite the useful features VGG16 already learned.

---

## Files in this Project

```
├── member1_dataset.py    → loads and prepares the dataset
├── member2_model.py      → VGG16 model and custom loss function
├── member3_train.py      → trains and saves the model
├── member4_predict.py    → evaluates and predicts fruit from image
└── README.md
```

---

## How to Run

**Install TensorFlow first:**
```
pip install tensorflow
```

**Step 1 — Set the dataset path**

Open `member1_dataset.py` and change line 14 to your dataset location:
```python
DATASET_PATH = r"C:\Users\YourName\Desktop\archive"
```

**Step 2 — Train the model**
```
python member3_train.py
```
This will train the model and save it as `fruit_model.keras`. Training may take 1–6 hours depending on your system. Using a GPU or Google Colab will be much faster.

**Step 3 — Test the model**
```
python member4_predict.py
```

**To predict a single image**, open `member4_predict.py` and set:
```python
TEST_IMAGE = r"C:\Users\YourName\Desktop\your_image.jpg" 
```

---

## Results

| | Value |
|--|--|
| Optimizer | Adam (lr = 0.0001) |
| Loss Function | Categorical Crossentropy |
| Epochs | 10 |
| Batch Size | 16 |
| Train / Val Split | 80% / 20% |
| Training Accuracy | *(fill after training)* |
| Validation Accuracy | *(fill after training)* | 

---

## References

- VGG16 Paper — Simonyan & Zisserman, 2015 — https://arxiv.org/abs/1409.1556
- Adam Optimizer Paper — Kingma & Ba, 2015 — https://arxiv.org/abs/1412.6980
- Dataset — https://www.kaggle.com/datasets/chrisfilo/fruit-recognition
- TensorFlow Docs — https://www.tensorflow.org

---

  Deep Learning Mini Project*
