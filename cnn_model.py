import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical
from keras.models import load_model
from tensorflow.keras import optimizers
from keras.utils import plot_model
import sys

# Load data and labels (sử dụng code của bạn để load dữ liệu)
dir = "./kyTu/kyTu/kyTu/"
categories = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]
data = []
labels = []

for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)

    for img in os.listdir(path):
        imgPath = os.path.join(path, img)
        kyTuImg = cv2.imread(imgPath, 0)

        image = np.array(kyTuImg)
        data.append(image)
        labels.append(label)

# Chuyển danh sách data và labels thành mảng numpy
data = np.array(data)
labels = np.array(labels)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Chuyển nhãn thành dạng one-hot
y_train = to_categorical(y_train, num_classes=len(categories))
y_test = to_categorical(y_test, num_classes=len(categories))

# Xây dựng mô hình CNN


def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(20, 20, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(len(categories), activation="softmax"))

    # Biên soạn mô hình
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def vgg_model():
    model = Sequential()
    model.add(
        Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=(20, 20, 1))
    )
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(Dense(4096, activation="relu"))
    model.add(Dense(len(categories), activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def test_model():
  # CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(20, 20, 1)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(34, activation='softmax'))

    model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(1e-3), metrics=['acc'])
    model.summary()
    plot_model(model, to_file="./model_structure.png", show_shapes=True, show_layer_names=True)
    return model

model = test_model()
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, verbose=1, )
cpt_save = ModelCheckpoint('./weight.h5', save_best_only=True, monitor='val_acc', mode='max')
#his = model.fit(X_train, y_train, validation_split=0.15, callbacks=[cpt_save, reduce_lr], epochs=500,
                #verbose=1, shuffle=True, batch_size=128, validation_data=(X_test, y_test))
#model.save("./model/cnn/model-500.keras")

def summarize_diagnostics(history):
    # plot loss
    plt.subplot(211)
    plt.title("Cross Entropy Loss")
    plt.plot(history.history["loss"], color="blue", label="train")
    plt.plot(history.history["val_loss"], color="orange", label="test")
    plt.legend()
    # plot accuracy
    plt.subplot(212)
    plt.title("Classification Accuracy")
    plt.plot(history.history["acc"], color="blue", label="train")
    plt.plot(history.history["val_acc"], color="orange", label="test")
    plt.legend()
    # save plot to file
    filename = sys.argv[0].split("/")[-1]
    plt.savefig(filename + "_plot500.png")
    plt.close()

#summarize_diagnostics(his)

