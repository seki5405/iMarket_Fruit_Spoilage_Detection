import argparse, sys, os
from pathlib import Path

import tensorflow as tf

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

    print(train_ds)
    for ds in train_ds:
        print(ds)

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
    
    return ds



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