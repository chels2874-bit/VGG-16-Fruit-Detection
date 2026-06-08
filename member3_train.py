from tensorflow.keras.optimizers import Adam

from member1_dataset import NUM_CLASSES, get_datasetss
from member2_model import build_vgg16_model, my_loss


EPOCHS =   10
MODEL_FILE = "fruit_model.keras"
CLASS_NAMES_FILE = "class_names.txt"


def save_class_names(class_names):
    
    file = open(CLASS_NAMES_FILE,"w")

    for name in class_names:
        file.write(name+"\n")

    file.close()

   print("Class names saved to", CLASS_NAMES_FILE)


def train():

    print("Loading dataset")
    train_data, val_data, class_names = get_datasets()

    print("Building model")
    model = build_vgg16_model(NUM_CLASSES)

    optimizer = Adam(learning_rate=0.0001)

    model.compile(
        optimizer = optimizer,
        loss = my_loss,
        metrics = ["accuracy"]
    )

    print("Training started this may take a while")
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS
    )

    model.save(MODEL_FILE)
    save_class_names(class_names)

    print("Training complete!")
    print("Model saved as:",MODEL_FILE)

train()

