# Basics
import pandas as pd
import argparse
from pathlib import Path

# Visualisation
import matplotlib.pyplot as plt

# Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


def print_training_info(arg):
    print("Running with: ")
    print(f"TF version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    print("")
    print("Training info:\n")
    print(f"Network architecture: {arg.net_model}")
    print(f"Input image size: {arg.input_width}x{arg.input_height}")
    print(f"Epochs : {arg.epochs}")
    print(f"Batch size : {arg.batch_size}")
    print(f"Validation split : {arg.val_split}")
    print(f"Input color mode : {arg.input_color_mode}")
    print(f"Path to save training stats: {arg.path_to_save_train_stats}")


def allow_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def prepare_labels(train_labels_path):
    max_velocity = 130.0  # [Km]
    labels_path = Path(train_labels_path)
    labels = pd.read_csv(labels_path)[['img_name', 'steer', 'velocity']]
    # normalize speed
    labels['velocity_normalized'] = labels.loc[:, 'velocity'] / max_velocity

    return labels


class Generators:
    """
    Train, validation and test generators
    """

    def __init__(self, train_data_path, test_data_path, train_df, test_df, input_height, input_width,
                 batch_size, color_mode, val_split):
        self.batch_size = batch_size
        self.color_mode = color_mode
        self.input_height = input_height
        self.input_width = input_width
        if color_mode == 'grayscale':
            self.img_shape = (self.input_height, self.input_width, 1)
        else:
            self.img_shape = (self.input_height, self.input_width, 3)

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
        self.model = None
        self.optimizer = None
        self.loss = None
        self.training = None

    def create_improved_pilot_net(self):
        """
        Build CNN model using img_width, img_height from fields.
        https://www.researchgate.net/publication/334080652_Self-Driving_Car_Steering_Angle_Prediction_Based_On_Deep_Neural_Network_An_Example_Of_CarND_Udacity_Simulator
        """
        self.model = Sequential([
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
        self.loss = losses.MeanAbsoluteError()
        self.optimizer = Adam(lr=0.0001)
        metrics_to_monitor = [metrics.MeanAbsoluteError(), metrics.MeanSquaredError()]
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=metrics_to_monitor)
        self.model.summary()

    def create_pilot_net(self, scale):
        """
        Nvidia PilotNetsel
        paper : https://arxiv.org/pdf/2010.08776.pdf

        @param scale - resize the network, 1.0 -> paper, 0.5 -> half of neurons

        """

        self.model = Sequential([
            Conv2D(input_shape=(self.img_width, self.img_height, self.img_depth), filters=int(24 * scale),
                   kernel_size=(5, 5),
                   strides=(2, 2), activation='elu', padding='valid'),
            Conv2D(filters=int(36 * scale), kernel_size=(5, 5), strides=(2, 2), activation='elu', padding='valid'),
            Conv2D(filters=int(48 * scale), kernel_size=(5, 5), strides=(3, 2), activation='elu', padding='valid'),
            Conv2D(filters=int(64 * scale), kernel_size=(3, 3), strides=(1, 1), activation='elu', padding='valid'),
            Conv2D(filters=int(64 * scale), kernel_size=(3, 3), strides=(1, 1), activation='elu', padding='same'),
            Flatten(),
            Dense(int(1164 * scale), activation='elu'),
            Dense(int(100 * scale), activation='elu'),
            Dense(int(50 * scale), activation='elu'),
            Dense(int(10 * scale), activation='elu'),
            Dense(2)
        ])

        self.loss = losses.MeanSquaredError()
        self.optimizer = Adam(lr=0.001)
        metrics_to_monitor = [metrics.MeanAbsoluteError(), metrics.MeanSquaredError()]
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=metrics_to_monitor)
        self.model.summary()

    def train(self, epochs: int, verb: int):
        """
        Train the model
        @param epochs number of training epoch
        @param verb - level of verbosity of trainer
        """
        steps_per_epoch = self.generators.train_generator.n // self.generators.batch_size
        validation_steps = self.generators.valid_generator.n // self.generators.batch_size

        # Save the best model during the training
        checkpointer = ModelCheckpoint('model-{epoch:02d}-{val_loss:.4f}.hdf5',
                                       monitor='val_loss',
                                       verbose=verb,
                                       save_best_only=True,
                                       mode='min')

        # We'll stop training if no improvement after some epochs
        earlystopper = EarlyStopping(monitor='val_loss', patience=25, verbose=1, mode="min")
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                      verbose=1, mode='min', min_lr=0.00001)

        # Train
        self.training = self.model.fit(self.generators.train_generator,
                                       epochs=epochs,
                                       steps_per_epoch=steps_per_epoch,
                                       validation_data=self.generators.valid_generator,
                                       validation_steps=validation_steps,
                                       callbacks=[earlystopper, reduce_lr, checkpointer],
                                       verbose=verb
                                       )

    def plot_save_history(self, path: str, save=True):
        """
        Plot training history
        @param path - path to save train history
        @param save - if True, save image, False do not save
        """

        # Trained model analysis and evaluation
        f, ax = plt.subplots(1, 1, figsize=(12, 3))
        ax.plot(self.training.history['loss'], label="Loss")
        ax.plot(self.training.history['val_loss'], label="Validation loss")
        ax.set_title('Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        if save:
            plt.savefig(path)
        plt.show()


def main(arg):
    labels = prepare_labels(arg.train_labels_path)

    # Create generators
    generators = Generators(train_data_path=arg.train_data_path,
                            test_data_path=None,
                            train_df=labels,
                            test_df=None,
                            input_height=arg.input_height,
                            input_width=arg.input_width,
                            batch_size=arg.batch_size,
                            color_mode=arg.input_color_mode,
                            val_split=arg.val_split)
    print("\nGenerators created !!")

    print_training_info(arg)
    input("Press any key to continue...")

    # Create and train the model
    trainer = ModelTrainer(generators)
    if arg.net_model == 'PilotNet':
        trainer.create_pilot_net(scale=1.0)
    elif arg.net_model == 'ImpPilotNet':
        trainer.create_improved_pilot_net()
    else:
        raise NotImplementedError("Choose from available models, check train_av.py --help")

    trainer.train(epochs=arg.epochs, verb=arg.verbose)

    trainer.plot_save_history(arg.path_to_save_train_stats)


if __name__ == "__main__":
    allow_memory_growth()

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, help='Path to train images', required=True)
    parser.add_argument('--train_labels_path', type=str, help='Path to train labels', required=True)
    parser.add_argument('--test_data_path', type=str, help='Path to test images')
    parser.add_argument('--input_height', type=int, help='Height of input image for CNN', default=264)
    parser.add_argument('--input_width', type=int, help='Width of input image for CNN', default=800)
    parser.add_argument('--val_split', type=float, help='Split between train and validation dataset', default=0.2)
    parser.add_argument('--epochs', type=int, help='Number of epochs of net training', default=2)
    parser.add_argument('--batch_size', type=int, help='Input batch size to network', default=128)
    parser.add_argument('--input_color_mode', type=str, help='rgb or grayscale input to network', default='rgb')
    parser.add_argument('--verbose', type=int, help='Verbosity of code', default=1)
    parser.add_argument('--net_model', type=str,
                        help='Choose between Improved PilotNet -> ImpPilotNet, Nvidia PilotNet -> PilotNet',
                        default='ImpPilotNet')
    parser.add_argument('--path_to_save_train_stats',
                        type=str, help="Path to save course of training", default='train_history.png')
    args, _ = parser.parse_known_args()
    main(args)
