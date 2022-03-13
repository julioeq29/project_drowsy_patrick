import time
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks
from tensorflow.keras.applications.vgg16 import VGG16
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from termcolor import colored

from project_drowsy.data import get_train_faces, get_test_faces, get_train_eyes, get_test_eyes
from project_drowsy.preprocessing import face_preprocess, eyes_preprocessing

from google.cloud import storage
from project_drowsy.params import *

# Mlflow wagon server
MLFLOW_URI = "https://mlflow.lewagon.co/"

class Trainer():
    # Mlflow paramters identifying the experiment.
    EXPERIMENT_NAME = "project_drowsy"

    def __init__(self, X_train_face, y_train_face, X_train_eye, y_train_eye, **kwargs):
        """

        """
        self.kwargs = kwargs
        # self.local = kwargs.get("local", True)
        self.X_train_face = np.array(X_train_face)
        self.y_train_face = np.array(y_train_face)
        self.X_train_eye = np.array(X_train_eye)
        self.y_train_eye = np.array(y_train_eye)
        self.mlflow = kwargs.get("mlflow", False)  # if True log info to nlflow
        self.experiment_name = kwargs.get("experiment_name", self.EXPERIMENT_NAME)


    def train(self):
        """
        This function calls the train_face_model and train_eye_model methods
        """
        self.train_face_model()
        self.train_eye_model()

    def train_face_model(self):
        tic = time.time()

        #train val split
        self.X_train_face, self.X_val_face, self.y_train_face, self.y_val_face = train_test_split(self.X_train_face, self.y_train_face, test_size=0.3)

        # Data augmentation
        train_generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
        val_generator = ImageDataGenerator(rescale=1/255)

        train_generator = train_generator.flow(self.X_train_face, self.y_train_face, shuffle=False)
        val_generator = val_generator.flow(self.X_val_face, self.y_val_face, shuffle=False)

        # Instanciate face model
        self.face_model = Sequential()

        self.face_model.add(Conv2D(256, (3, 3), activation="relu", input_shape=self.X_train_face.shape[1:]))
        self.face_model.add(MaxPooling2D(2, 2))

        self.face_model.add(Conv2D(128, (3, 3), activation="relu"))
        self.face_model.add(MaxPooling2D(2, 2))

        self.face_model.add(Conv2D(64, (3, 3), activation="relu"))
        self.face_model.add(MaxPooling2D(2, 2))

        self.face_model.add(Conv2D(32, (3, 3), activation="relu"))
        self.face_model.add(MaxPooling2D(2, 2))

        self.face_model.add(Flatten())
        self.face_model.add(Dropout(0.5))

        self.face_model.add(Dense(64, activation="relu"))
        self.face_model.add(Dense(1, activation="sigmoid"))

        self.face_model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")

        # Fit model
        es = callbacks.EarlyStopping(patience=2, restore_best_weights=True)
        self.face_model.fit(train_generator,
                            epochs=15,
                            validation_data=val_generator,
                            shuffle=True,
                            validation_steps=len(val_generator),
                            callbacks=[es],
                            verbose=1)
        # mlflow log training time
        self.mlflow_log_metric("face_train_time", int(time.time() - tic))

    def train_eye_model(self):
        tic = time.time()

        #train val split
        self.X_train_eye, self.X_val_eye, self.y_train_eye, self.y_val_eye = train_test_split(self.X_train_eye, self.y_train_eye, test_size=0.3)

        #Create image generator for augmentation
        train_generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
        val_generator = ImageDataGenerator(rescale=1/255)

        train_generator = train_generator.flow(np.array(self.X_train_eye), self.y_train_eye, shuffle=False)
        val_generator = val_generator.flow(np.array(self.X_val_eye), self.y_val_eye, shuffle=False)

        #Instanciate model
        eye_model = Sequential()

        eye_model.add(Conv2D(256, (3, 3), activation="relu", input_shape=self.X_train_eye.shape[1:]))
        eye_model.add(MaxPooling2D(2, 2))

        eye_model.add(Conv2D(128, (3, 3), activation="relu"))
        eye_model.add(MaxPooling2D(2, 2))

        eye_model.add(Conv2D(64, (3, 3), activation="relu"))
        eye_model.add(MaxPooling2D(2, 2))

        eye_model.add(Conv2D(32, (3, 3), activation="relu"))
        eye_model.add(MaxPooling2D(2, 2))

        eye_model.add(Flatten())
        eye_model.add(Dropout(0.5))

        eye_model.add(Dense(64, activation="relu"))
        eye_model.add(Dense(1, activation="sigmoid"))

        eye_model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")

        #fit
        es = callbacks.EarlyStopping(patience=5, restore_best_weights=True)

        eye_model.fit(train_generator,
                      #batch_size=16,
                      epochs=15,
                      validation_data=val_generator,
                      shuffle=True,
                      validation_steps=len(val_generator),
                      callbacks=[es],
                      verbose=1)

        # mlflow log training time
        self.mlflow_log_metric("eye_train_time", int(time.time() - tic))

        self.eye_model = eye_model

    def evaluate(self, X_test_face, y_test_face, X_test_eye, y_test_eye):
        self.evaluate_face_model(X_test_face, y_test_face)
        self.evaluate_eye_model(X_test_eye, y_test_eye)

    def evaluate_face_model(self, X_test_face, y_test_face):
        # mlfow log train accuracy
        accuracy_face_train = self.face_model.evaluate(self.X_train_face, self.y_train_face)[1]
        self.mlflow_log_metric("accuracy_face_train", accuracy_face_train)
        # mlfow log test accuracy
        accuracy_face_val = self.face_model.evaluate(self.X_val_face, self.y_val_face)[1]
        self.mlflow_log_metric("accuracy_face_val", accuracy_face_val)
        # mlfow log test accuracy
        X_test_face = np.array(X_test_face)
        y_test_face = np.array(y_test_face)
        accuracy_face_test = self.face_model.evaluate(X_test_face, y_test_face)[1]
        self.mlflow_log_metric("accuracy_face_test", accuracy_face_test)
        return accuracy_face_test

    def evaluate_eye_model(self,X_test_eye,y_test_eye):
        # mlflow log train accuracy
        accuracy_eye_train = self.eye_model.evaluate(self.X_train_eye, self.y_train_eye)[1]
        self.mlflow_log_metric("accuracy_eye_train", accuracy_eye_train)
        # mlfow log validation accuracy
        accuracy_eye_val = self.eye_model.evaluate(self.X_val_eye, self.y_val_eye)[1]
        self.mlflow_log_metric("accuracy_eye_val", accuracy_eye_val)
        # mlfow log test accuracy
        X_test_eye = np.array(X_test_eye)
        y_test_eye = np.array(y_test_eye)
        accuracy_eye_test = self.eye_model.evaluate(X_test_eye, y_test_eye)[1]
        self.mlflow_log_metric("accuracy_eye_test", accuracy_eye_test)
        print(colored(f"accuracy train: {accuracy_eye_train} || accuracy val: {accuracy_eye_val} || accuracy test: {accuracy_eye_test}"), "blue")
        return accuracy_eye_test

    def save_models(self):
        self.save_face_model()
        self.save_eye_model()

    def save_face_model(self):
        """Save the face model into a .joblib and upload it on Google Storage
        / models folder"""
        # joblib.dump(self.face_model, 'project_drowsy/models/face_model.joblib')
        self.face_model.save('project_drowsy/models/face_model.h5')

        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob('project_drowsy/models/face_model.h5')
        blob.upload_from_filename('project_drowsy/models/face_model.h5')
        print(
            f"uploaded model to gcp cloud storage under \n => 'project_drowsy/models/face_model.h5'"
        )
    def save_eye_model(self):
        self.eye_model.save('project_drowsy/models/eye_model.h5')

        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob('project_drowsy/models/eye_model.h5')
        blob.upload_from_filename('project_drowsy/models/eye_model.h5')
        print(
            f"uploaded model to gcp cloud storage under \n => 'project_drowsy/models/eye_model.h5'"
        )
    ## MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":

    experiment = "project_drowsy"
    data_params = dict(
                #n_images=500,
                  local=False,
                  mlflow=True,
                  experiment_name=experiment,
                  random_seed=1)



    # Load and preprocess train face data
    print(colored("############   Loading Train Face Data   ############", "blue"))
    face_train_images = get_train_faces(**data_params)
    preprocess_params = dict(file_path=TRAIN_PATH,
                             local=False,
                             images=face_train_images,
                             predict=False)
    train_face_data = face_preprocess(**preprocess_params)
    X_train_face = []
    y_train_face = []
    for image, label in train_face_data:
        X_train_face.append(image)
        y_train_face.append(label)



    # Load and preprocess test face data
    print(colored("############   Loading Test Face Data   ############", "blue"))
    face_test_images = get_test_faces(**data_params)
    preprocess_params = dict(file_path=TEST_PATH,
                             local=False,
                             images=face_test_images,
                             predict=False)
    test_face_data = face_preprocess(**preprocess_params)
    X_test_face = []
    y_test_face = []
    for image, label in test_face_data:
        X_test_face.append(image)
        y_test_face.append(label)



    # Load and preprocess train eye data
    print(colored("############   Loading Train Eye Data   ############", "blue"))
    eye_train_images = get_train_eyes(**data_params)
    preprocess_params = dict(file_path=TRAIN_PATH,
                             local=False,
                             images=eye_train_images,
                             predict=False,
                             image_size=145)

    X_train_eye, y_train_eye = eyes_preprocessing(**preprocess_params)

    # Get and preprocess test eyes images
    print(colored("############   Loading Test Eye Data   ############", "blue"))
    eye_test_images = get_test_eyes(**data_params)
    preprocess_params = dict(file_path=TEST_PATH,
                             local=False,
                             images=eye_test_images,
                             predict=False,
                             image_size=145)
    X_test_eye, y_test_eye = eyes_preprocessing(**preprocess_params)




    #Train models
    t = Trainer(X_train_face, y_train_face, X_train_eye, y_train_eye,
                **data_params)
    print(t.X_train_face)
    print(colored("############  Training both models and logging training time to mlflow  ############", "red"))
    t.train()
    print(colored("############  Evaluating both models and logging accuracy to mlflow  ############", "blue"))
    t.evaluate(X_test_face, y_test_face, X_test_eye, y_test_eye)
    print(colored("############   Saving model    ############", "green"))
    t.save_models()
