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
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

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


def print_trainnig_stats(args):
    print("Running with: ")
    print(f"TF version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    print("")
    print("Training stats:\n")
    print(f"Epochs : {args.epochs}")
    print(f"Batch size : {args.batch_size}")
    print(f"Validation split : {args.val_split}")
    print(f"Input color mode : {args.input_color_mode}")


class Generators:
    """
    Train, validation and test generators
    """

    def __init__(self, train_data_path, test_data_path, train_df, test_df, batch_size, color_mode, val_split):
        self.batch_size = batch_size
        self.color_mode = color_mode
        if color_mode == 'grayscale':
            self.img_shape = (200, 200, 1)
        else:
            self.img_shape = (200, 200, 3)

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

    def create_model_from_article(self):
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

    def create_model(self,
                     kernel_size=(3, 3),
                     pool_size=(2, 2),
                     first_filters=32,
                     second_filters=64,
                     third_filters=128,
                     first_dense=256,
                     second_dense=128,
                     dropout_conv=0.3,
                     dropout_dense=0.3):
        """
        NOT IMPLEMENTET WELL
        """

        model = Sequential()
        # First conv filters
        model.add(Conv2D(first_filters, kernel_size, activation='relu', padding="same",
                         input_shape=(self.img_width, self.img_height, 3)))
        model.add(Conv2D(first_filters, kernel_size, padding="same", activation='relu'))
        model.add(Conv2D(first_filters, kernel_size, padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(dropout_conv))

        # Second conv filter
        model.add(Conv2D(second_filters, kernel_size, padding="same", activation='relu'))
        model.add(Conv2D(second_filters, kernel_size, padding="same", activation='relu'))
        model.add(Conv2D(second_filters, kernel_size, padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(dropout_conv))

        # Third conv filter
        model.add(Conv2D(third_filters, kernel_size, padding="same", activation='relu'))
        model.add(Conv2D(third_filters, kernel_size, padding="same", activation='relu'))
        model.add(Conv2D(third_filters, kernel_size, padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(dropout_conv))

        model.add(Flatten())

        # First dense
        model.add(Dense(first_dense, activation="relu"))
        model.add(Dropout(dropout_dense))
        # Second dense
        model.add(Dense(second_dense, activation="relu"))
        model.add(Dropout(dropout_dense))

        # Out layer
        model.add(Dense(2, activation="softmax"))

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


class Evaluator:
    """
    Evaluaion :predict on test data (not submission data from test folder)
    and print reports, plot results etc.
    """

    def __init__(self, model, training, generator, y_true):
        self.training = training
        self.generator = generator
        # predict the data
        steps = 5
        self.y_pred_raw = model.predict_generator(self.generator, steps=steps)
        self.y_pred = np.argmax(self.y_pred_raw, axis=1)
        self.y_true = y_true[:len(self.y_pred)]

    """
    Accuracy, evaluation
    """

    def plot_history(self):
        """
        Plot training history
        """
        ## Trained model analysis and evaluation
        f, ax = plt.subplots(1, 2, figsize=(12, 3))
        ax[0].plot(self.training.history['loss'], label="Loss")
        ax[0].plot(self.training.history['val_loss'], label="Validation loss")
        ax[0].set_title('Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()

        # Accuracy
        ax[1].plot(self.training.history['acc'], label="Accuracy")
        ax[1].plot(self.training.history['val_acc'], label="Validation accuracy")
        ax[1].set_title('Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()
        plt.tight_layout()
        plt.show()

    def plot_roc(self):
        # y_pred_keras = model.predict_generator(test_gen, steps=len(df_val), verbose=1)
        # Calculate roc
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(self.y_true, self.y_pred)
        auc_keras = auc(fpr_keras, tpr_keras)

        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_keras, tpr_keras, label='area = {:.3f}'.format(auc_keras))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()

    def print_report(self):
        """
        Predict and evaluate using ground truth from labels
        Test generator did not shuffle 
        and we can use true labels for comparison
        """
        # Print classification report
        print(metrics.classification_report(self.y_true, self.y_pred))


def preapare_labels(train_labels_path):
    first_labels_path = Path(train_labels_path)
    second_labels_path = Path(train_labels_path)

    first_labels = pd.read_csv(first_labels_path)
    second_labels = pd.read_csv(second_labels_path)

    first_labels_useful = first_labels[['img_name', 'steer', 'velocity']]
    second_labels_useful = second_labels[['img_name', 'steer', 'velocity']]
    # TODO merge datasets
    # normalize speed
    MAX_VELOCITY = 130.0  # [Km]
    first_labels_useful['velocity_normalized'] = first_labels_useful.loc[:, 'velocity'] / MAX_VELOCITY
    second_labels_useful['velocity_normalized'] = second_labels_useful.loc[:, 'velocity'] / MAX_VELOCITY

    return first_labels_useful


def main(args):
    labels = preapare_labels(args.train_labels_path)
    # Create generators        
    generators = Generators(train_data_path=args.train_data_path,
                            test_data_path=None,
                            train_df=labels,
                            test_df=None,
                            batch_size=args.batch_size,
                            color_mode=args.input_color_mode,
                            val_split=args.val_split)
    print("\nGenerators created !!")

    print_trainnig_stats(args)

    input("Press any key to continue...")
    # Create and train the model
    trainer = ModelTrainer(generators)
    model = trainer.create_model_from_article()
    model.summary()
    # print(type(model))
    training = trainer.train(model=model,
                             epochs=args.epochs,
                             verb=args.verbose)

    # Create evaluator instance
    # evaluator = Evaluator(model, training, generators.test_generator, test_df.label.values)

    # Draw accuracy and loss charts
    # evaluator.plot_history()

    # ROC curve
    # evaluator.plot_roc()

    # Classification report
    # evaluator.print_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, help='Path to train images', required=True)
    parser.add_argument('--train_labels_path', type=str, help='Path to train labels', required=True)
    parser.add_argument('--test_data_path', type=str, help='Path to test images')
    parser.add_argument('--val_split', type=float, help='Split between train and validation dataset', default=0.2)
    parser.add_argument('--epochs', type=int, help='Number of epochs of net training', default=2)
    parser.add_argument('--batch_size', type=int, help='Input batch size to network', default=128)
    parser.add_argument('--input_color_mode', type=str, help='rgb or grayscale input to network', default='grayscale')
    parser.add_argument('--verbose', type=int, help='Verbosity of code', default=1)
    args, _ = parser.parse_known_args()
    main(args)
