"""CNN-based image classification on SageMaker with TensorFlow and Keras

Reference solution implementation modified for pipe mode to investigate:
https://github.com/aws/sagemaker-tensorflow-extensions/issues/46
"""

# Dependencies:
import argparse
import json
import os

import numpy as np
from PIL import Image
from sagemaker_tensorflow import PipeModeDataset
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

def parse_args():
    """Acquire hyperparameters and directory locations passed by SageMaker"""
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)

    # Data, model, and output directories
    hps = json.loads(os.environ.get("SM_HPS", {}))
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])

    parser.add_argument("--num-samples-train", type=int, default=hps.get("num-samples-train"))
    parser.add_argument("--num-samples-test", type=int, default=hps.get("num-samples-test"))

    return parser.parse_known_args()

# TODO: Take number of total digits and image dimensions as params when the basics are working.
def tf_mapper(fields):
    img = tf.image.decode_image(fields[0], dtype=tf.float32, channels=1) / 255.
    # MNIST = all images already prepared to 28px square
    img.set_shape((28, 28, 1))
    
    digit = tf.strings.to_number(fields[1], out_type=tf.int32)
    digit.set_shape([])
    digit_onehot = tf.one_hot(digit, 10)
    digit_onehot.set_shape([10,])

    return img, digit_onehot


def load_data(args):
    ds_train = PipeModeDataset(channel="train") \
        .repeat(args.epochs) \
        .batch(2) \
        .map(tf_mapper) \
        .batch(args.batch_size, drop_remainder=True)

    ds_test = PipeModeDataset(channel="test") \
        .repeat(args.epochs) \
        .batch(2) \
        .map(tf_mapper) \
        .batch(args.batch_size, drop_remainder=True)

    return ds_train, ds_test


def build_model(input_shape, n_labels):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(n_labels, activation="softmax"))

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adadelta(),
        metrics=["accuracy"]
    )

    return model


# Training script:
if __name__ == "__main__":
    args, _ = parse_args()
    print(args)

    ds_train, ds_test = load_data(args)

    model = build_model((28, 28, 1), 10)

    model.fit(
        ds_train,
        epochs=args.epochs,
        verbose=2,
        shuffle=False,
        steps_per_epoch=args.num_samples_train // args.batch_size,
        validation_data=ds_test,
        validation_steps=args.num_samples_test // args.batch_size,
    )

    # Save outputs (trained model) to specified folder in TFServing-compatible format
    sess = K.get_session()
    tf.saved_model.simple_save(
        sess,
        os.path.join(args.model_dir, "model/1"),
        inputs={ "inputs": model.input },
        outputs={ t.name: t for t in model.outputs },
    )
