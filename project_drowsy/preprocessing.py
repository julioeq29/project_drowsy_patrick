import numpy as np
import cv2
import mediapipe as mp
import numpy as np
import os
from project_drowsy.utils import getLandmarks, getLeftEyeRect, getRightEyeRect
from project_drowsy.params import *
# from google.cloud import storage


def face_preprocess(**params):
    if params['predict']:
        image = params['webcam']
        if (image is not None) or (params is not None):
          single_cropped_face, coords = detect_face(image, **params)
          reshape_img = np.reshape(single_cropped_face, [1,145,145,3])
          # reshape_img = np.expand_dims(single_cropped_face, axis=0)
          return reshape_img, coords

    face_data = []
    categories = ["yawn", "no_yawn"]
    client = storage.Client()
    source_bucket = client.get_bucket(BUCKET_NAME)

    for category in categories:
        path_link = os.path.join(params['file_path'], category)
        print(path_link)
        class_num = categories.index(category)
        images = params['images']
        for image in images[category]:

            if params['local']:
                image = cv2.imread(f"{path_link}/{image}")

            else:
                source_blob = source_bucket.get_blob(
                    f'project_drowsy/{path_link}/{image}')
                print(source_blob)
                image = np.asarray(bytearray(source_blob.download_as_string()),
                                   dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            resized_image = detect_face(image, image_size=145)
            if resized_image is not None:
                face_data.append([resized_image, class_num])

    return face_data


def detect_face(image, image_size=145, **params):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cant_find = []
    # If no face detections found, skip
    if not results.detections:
        cant_find.append([image])
        return None

    # create a copy of the image to draw on
    annotated_image = image.copy()

    # get the coordinates of the bounding box from the detection object
    xmin = results.detections[0].location_data.relative_bounding_box.xmin
    ymin = results.detections[0].location_data.relative_bounding_box.ymin
    width = results.detections[0].location_data.relative_bounding_box.width
    height = results.detections[0].location_data.relative_bounding_box.height

    x = int(xmin * annotated_image.shape[1])
    y = int(ymin * annotated_image.shape[0])
    w = int(width * annotated_image.shape[1])
    h = int(height * annotated_image.shape[0])

    # crop the image
    cropped_image = annotated_image[y:y + h, x:x + w]

    # resize image to standard size
    try:
        resized_image = cv2.resize(cropped_image, (image_size, image_size))
        coords = (x,x+w,y,y+h)
        if params['predict']:
            return resized_image, coords

        return resized_image
    except:
        pass


def eyes_preprocessing(**params):
    # for real webcam data

    if params['predict']:

        image = params['webcam']
        eyes, eye_coords = detect_eyes(image, **params)
        left_eye, right_eye = eyes
        left_eye = (np.reshape(left_eye, [1,145,145,3]),eye_coords[0])

        # left_eye =  np.expand_dims(left_eye, axis=0)
        right_eye = (np.reshape(right_eye, [1,145,145,3]),eye_coords[1])
        # right_eye =  np.expand_dims(right_eye, axis=0)

        return left_eye, right_eye



    #For train or test data only
    eye_images = []
    eye_classes = []
    categories = ["Closed", "Open"]
    client = storage.Client()
    source_bucket = client.get_bucket(BUCKET_NAME)

    for category in categories:
        path_link = os.path.join(params['file_path'], category)
        class_num = categories.index(category)
        images = params['images']
        for image in images[category]:
            image_name = image
            if params['local']:
                image = cv2.imread(f"{path_link}/{image}")
            else:
                source_blob = source_bucket.get_blob(f'project_drowsy/{path_link}/{image}')
                image = np.asarray(bytearray(source_blob.download_as_string()),
                                   dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            resized_image = cv2.resize(image, (image_size, image_size))
            if np.array(resized_image).shape==(145,145,4):
                print(category)
                print(image_name)
            eye_images.append(resized_image)
            eye_classes.append(class_num)

    return eye_images, eye_classes



def detect_eyes(image, image_size=145, **params):

    ''' Return two cropped eye images from a single image
    and stores them in a list of two elelments'''
    eyes = []
    try:
        landmarks = getLandmarks(image)

        # RIGHT EYE
        x_r, y_r, w_r, h_r = getRightEyeRect(image, landmarks)

        # LEFT EYE
        x_l, y_l, w_l, h_l = getLeftEyeRect(image, landmarks)

        # pad and crop left eye
        x_sf = 0.3
        y_sf = 1
        x_pad_min_l = x_l - int(w_l * x_sf)
        x_pad_max_l = x_l + w_l + int(w_l * x_sf)
        y_pad_min_l = y_l - int(h_l * y_sf)
        y_pad_max_l = y_l + h_l + int(h_l * y_sf)
        eye_l = image[y_pad_min_l:y_pad_max_l, x_pad_min_l:x_pad_max_l]
        eye_l = cv2.resize(eye_l, (image_size, image_size))
        eyes.append(eye_l)

        # pad and crop right eye
        x_sf = 0.3
        y_sf = 1
        x_pad_min_r = x_r - int(w_r * x_sf)
        x_pad_max_r = x_r + w_r + int(w_r * x_sf)
        y_pad_min_r = y_r - int(h_r * y_sf)
        y_pad_max_r = y_r + h_r + int(h_r * y_sf)
        eye_r = image[y_pad_min_r:y_pad_max_r, x_pad_min_r:x_pad_max_r]
        eye_r = cv2.resize(eye_r, (image_size, image_size))
        eyes.append(eye_r)

    except Exception as e:
        print(e)

    if params['predict']:
        coords_r = (x_pad_min_r, x_pad_max_r, y_pad_min_r, y_pad_max_r)
        coords_l = (x_pad_min_l, x_pad_max_l, y_pad_min_l, y_pad_max_l)
        coords = (coords_l,coords_r)
        return eyes,coords

    return eyes


if __name__ == '__main__':
    pass
