import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from project_drowsy.data import get_test_faces, get_test_eyes
from project_drowsy.preprocessing import face_preprocess, eyes_preprocessing
# from project_drowsy.trainer import evaluate

BUCKET_NAME = 'wagon-data-753-rigg'
MODEL_NAME = 'project_drowsy'
BUCKET_TRAIN_DATA_PATH = 'data/raw_data/train'


def get_img():
    img = f"{os.getcwd()}/project_drowsy/data/images/Frame4.jpg"
    img = cv2.imread(img)
    return img

def get_models():
    local_path = f"{os.getcwd()}/project_drowsy/models/"
    face_model = load_model(os.path.join(local_path, 'project_drowsy_models_face_model.h5'))
    eye_model = load_model(os.path.join(local_path, 'project_drowsy_models_eye_model.h5'))
    return face_model, eye_model

def mapping(face_predict, left_eye_predict, right_eye_predict):
    result = []
    if face_predict == 0:
        result.append('yawn')
    else:
        result.append('no_yawn')

    if left_eye_predict == 0:
        result.append('closed')
    else:
        result.append('open')

    if right_eye_predict == 0:
        result.append('closed')
    else:
        result.append('open')

    return result

def make_prediction(**params):
    cropped_face, face_coords = face_preprocess(**params)
    cropped_left_eye, cropped_right_eye = eyes_preprocessing(**params)
    return cropped_face, face_coords, cropped_left_eye, cropped_right_eye
    # face_model, eye_model = get_models()

    # yawn_prob = face_model.predict(cropped_face)[0][0]
    # left_eye_prob = eye_model.predict(cropped_left_eye[0])[0][0]
    # right_eye_prob = eye_model.predict(cropped_right_eye[0])[0][0]
    # probs = [yawn_prob, left_eye_prob, right_eye_prob]

    # yawn_prediction = (yawn_prob > 0.5).astype("int32")
    # left_eye_prediction = (left_eye_prob > 0.5).astype("int32")
    # right_eye_prediction = (right_eye_prob > 0.5).astype("int32")
    # prediction = mapping(yawn_prediction, left_eye_prediction, right_eye_prediction)

    # return prediction, probs, face_coords, cropped_left_eye[1], cropped_right_eye[1]


if __name__ == '__main__':
    image = get_img()
    predictions = make_prediction(image)
    print(predictions)
