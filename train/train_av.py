# Basics
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Charts
import matplotlib.pyplot as plt
import seaborn as sns

# ML, statistics
import scipy
from sklearn import metrics

# Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


def print_trainnig_info(args):
    print("Running with: ")
    print(f"TF version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    print("")
    print("Training stats:\n")
    print(f"Input image size: {args.input_shape}x{args.input_shape}")
    print(f"Epochs : {args.epochs}")
    print(f"Batch size : {args.batch_size}")
    print(f"Validation split : {args.val_split}")
    print(f"Input color mode : {args.input_color_mode}")


class Generators:
    """
    Train, validation and test generators
    """

    def __init__(self, train_data_path, test_data_path, train_df, test_df, input_shape, batch_size, color_mode, val_split):
        self.batch_size = batch_size
        self.color_mode = color_mode
        self.input_shape = input_shape
        if color_mode == 'grayscale':
            self.img_shape = (self.input_shape, self.input_shape, 1)
        else:
            self.img_shape = (self.input_shape, self.input_shape, 3)

        # Base train/validation generator
        _datagen = ImageDataGenerator(
            rescale=1. / 255.,
            validation_split=val_split
        )
        # Train generator
        self.train_generator = _datagen.flow_from_dataframe(
            dataframe=train_df,
            directory=train_data_path,
            subset='training',
            x_col="img_name",
            y_col=['steer', 'velocity_normalized'],
            class_mode="raw",
            shuffle=True,
            target_size=(self.img_shape[0], self.img_shape[1]),
            color_mode=self.color_mode,
            batch_size=self.batch_size)
        print('Train generator created\n')

        # Validation generator
        self.valid_generator = _datagen.flow_from_dataframe(
            dataframe=train_df,
            directory=train_data_path,
            subset='validation',
            x_col="img_name",
            y_col=['steer', 'velocity_normalized'],
            class_mode="raw",
            shuffle=True,
            target_size=(self.img_shape[0], self.img_shape[1]),
            color_mode=self.color_mode,
            batch_size=self.batch_size)
        print('Validation generator created\n')

        # Test generator
        if test_df is not None:
            _test_datagen = ImageDataGenerator(rescale=1. / 255.)
            self.test_generator = _test_datagen.flow_from_dataframe(
                dataframe=test_df,
                directory=test_data_path,
                x_col="img_name",
                y_col=['steer', 'velocity_normalized'],
                class_mode="raw",
                batch_size=self.batch_size,
                seed=42,
                shuffle=False,
                target_size=(self.img_shape[0], self.img_shape[1]))
            print('Test generator created\n')
        else:
            print('Test generator not created, no test_df sourced..')


class ModelTrainer:
    """
    Create and fit the model
    """

    def __init__(self, generators):
        self.generators = generators
        self.img_width = generators.img_shape[0]
        self.img_height = generators.img_shape[1]
        self.img_depth = generators.img_shape[2]

    def create_model(self):
        """
        Build CNN model using img_width, img_height from fields.
        https://www.researchgate.net/publication/334080652_Self-Driving_Car_Steering_Angle_Prediction_Based_On_Deep_Neural_Network_An_Example_Of_CarND_Udacity_Simulator
        """
        model = Sequential([
            Conv2D(input_shape=(self.img_width, self.img_height, self.img_depth), filters=8, kernel_size=(9, 9),
                   strides=(3, 3), activation='elu', padding='valid'),
            BatchNormalization(),
            Conv2D(filters=16, kernel_size=(5, 5), strides=(3, 3), activation='elu', padding='valid'),
            BatchNormalization(),
            Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='elu', padding='valid'),
            BatchNormalization(),
            Dropout(rate=0.2),
            Flatten(),
            Dense(50, activation='elu'),
            Dense(2, activation='linear')
        ])
        loss = losses.MeanAbsoluteError()
        optimizer = Adam(lr=0.0001)
        metrics_to_monitor = [metrics.MeanAbsoluteError(), metrics.MeanSquaredError()]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics_to_monitor)

        return model

    def train(self, model, epochs: int, verb: int):
        """
        Train the model
        """
        steps_per_epoch = self.generators.train_generator.n // self.generators.batch_size
        validation_steps = self.generators.valid_generator.n // self.generators.batch_size

        # Save the best model during the traning
        checkpointer = ModelCheckpoint('best_model1.h5',
                                       monitor='val_mean_absolute_error',
                                       verbose=verb,
                                       save_best_only=True,
                                       mode='min')

        # We'll stop training if no improvement after some epochs
        earlystopper = EarlyStopping(monitor='val_mean_absolute_error', patience=10, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.5, patience=5,
                                      verbose=1, mode='min', min_lr=0.00001)

        # Train
        training = model.fit(self.generators.train_generator,
                             epochs=epochs,
                             steps_per_epoch=steps_per_epoch,
                             validation_data=self.generators.valid_generator,
                             validation_steps=validation_steps,
                             callbacks=[earlystopper, reduce_lr, checkpointer],
                             verbose=verb
                             )
        # Get the best saved weights
        # model.load_weights('best_model1.h5')
        return training


def preapare_labels(train_labels_path):
    MAX_VELOCITY = 130.0  # [Km]
    labels_path = Path(train_labels_path)
    labels = pd.read_csv(labels_path)[['img_name', 'steer', 'velocity']]
    # normalize speed
    labels['velocity_normalized'] = labels.loc[:, 'velocity'] / MAX_VELOCITY

    return labels


def main(args):
    labels = preapare_labels(args.train_labels_path)
    print(labels)

    # Create generators
    generators = Generators(train_data_path=args.train_data_path,
                            test_data_path=None,
                            train_df=labels,
                            test_df=None,
                            input_shape=args.input_shape,
                            batch_size=args.batch_size,
                            color_mode=args.input_color_mode,
                            val_split=args.val_split)
    print("\nGenerators created !!")

    print_trainnig_info(args)
    input("Press any key to continue...")

    # Create and train the model
    trainer = ModelTrainer(generators)
    model = trainer.create_model_from_article()
    model.summary()
    training = trainer.train(model=model,
                             epochs=args.epochs,
                             verb=args.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, help='Path to train images', required=True)
    parser.add_argument('--train_labels_path', type=str, help='Path to train labels', required=True)
    parser.add_argument('--test_data_path', type=str, help='Path to test images')
    parser.add_argument('--input_shape', type=int, help='Shape of input image for CNN', default=200)
    parser.add_argument('--val_split', type=float, help='Split between train and validation dataset', default=0.2)
    parser.add_argument('--epochs', type=int, help='Number of epochs of net training', default=2)
    parser.add_argument('--batch_size', type=int, help='Input batch size to network', default=128)
    parser.add_argument('--input_color_mode', type=str, help='rgb or grayscale input to network', default='rgb')
    parser.add_argument('--verbose', type=int, help='Verbosity of code', default=1)
    args, _ = parser.parse_known_args()
    main(args)
