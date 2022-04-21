import argparse, sys, os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import VGG16, MobileNetV2
from keras.models import Sequential
from keras.layers import *
import matplotlib.pyplot as plt

def main(opt):
    print(opt)
    base_model = opt.base_model
    epochs = opt.epochs
    batch_size = opt.batch_size
    optimizer = opt.optimizer
    weight_name = opt.weight_name
    dataset = opt.dataset
    imgsz = opt.imgsz
    split = opt.split

    dataset_path = dataset
    BATCH_SIZE = batch_size
    IMG_SIZE = (imgsz, imgsz)

    train_ds = get_dataset(dataset_path, 123, split, 'training', IMG_SIZE, BATCH_SIZE)
    val_ds = get_dataset(dataset_path, 123, split, 'validation', IMG_SIZE, BATCH_SIZE)

    model = get_model(base_model)
    print(model.summary())

    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    plot_show(hist)

    show_example(model, val_ds)





    

def get_dataset(dataset_path, seed, split, subset, img_size, batch_size):
    ds = tf.keras.preprocessing.image_dataset_from_directory(dataset_path,
                                                             seed=seed,
                                                             validation_split=split,
                                                             subset=subset,
                                                             image_size=img_size,
                                                             batch_size=batch_size)
    def preprocess(img, ans):
        return img/255., float(ans)

    ds = ds.map(preprocess)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    
    return ds

def get_model(base_model):
    input_shape=(224,224,3)
    if base_model == 'vgg16':
        base = VGG16(weights='imagenet', input_shape=input_shape, include_top=False)
    elif base_model == 'mobilenet':
        base = MobileNetV2(weights='imagenet', input_shape=input_shape, include_top=False)

    model = Sequential()
    model.add(base)
    model.add(Flatten())
    model.add(Dense(1596, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(796, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(56, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='linear'))
    for layer in model.layers[:-10]:
        layer.trainable = False

    opt = tf.keras.optimizers.Adam(lr=1e-5, decay=1e-3 / 200)
    model.compile(loss="mean_squared_error", optimizer=opt)

    return model


def plot_show(history):
  plt.plot(history["accuracy"], color ="blue", label = "accuracy")
  plt.plot(history["loss"], color="red", label="loss")

  plt.title(f"Curves of {base_model} based regression model") 
#   plt.ylabel("Accuracy(0~1)")
  plt.xlabel("Number of epochs")
  plt.legend()
  plt.show()

def show_example(model, ds):
    test_img = [ds[0][:5] for ds in ds.take(1)]
    test_lb = [ds[1][:5] for ds in ds.take(1)]

    pred = model.predict(test_img)

    for idx, img in enumerate(test_img[0]):
        plt.imshow(img)
        title = "Pred : " + round(pred[idx], 2) + "GT : " + test_lb[idx]
        plt.title(title)
        plt.show()

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', type=str, default='vgg16', help='Base model for the regression model')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Training batch size')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='Adam', help='optimizer')
    parser.add_argument('--weight-name', type=str, required=True, help='Name to save weights after training')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='Image size (width = height)')
    parser.add_argument('--split', type=float, default=0.2, help='train_valid split ratio')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLOv5 root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

    print(ROOT, "LOOK!!")
    opt = parse_opt()
    main(opt)