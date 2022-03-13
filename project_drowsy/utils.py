import mediapipe as mp

def getLandmarks(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_mesh.process(image)
    landmarks = results.multi_face_landmarks[0].landmark
    return landmarks

def getRightEyeRect(image, landmarks):
    eye_top = int(landmarks[257].y * image.shape[0])
    eye_left = int(landmarks[362].x * image.shape[1])
    eye_bottom = int(landmarks[374].y * image.shape[0])
    eye_right = int(landmarks[263].x * image.shape[1])

    cloned_image = image.copy()
    cropped_right_eye = cloned_image[eye_top:eye_bottom, eye_left:eye_right]
    h, w, _ = cropped_right_eye.shape
    x = eye_left
    y = eye_top
    return x, y, w, h

def getLeftEyeRect(image, landmarks):
    # eye_left landmarks (27, 23, 130, 133) ->? how to utilize z info
    eye_top = int(landmarks[159].y * image.shape[0])
    eye_left = int(landmarks[33].x * image.shape[1])
    eye_bottom = int(landmarks[145].y * image.shape[0])
    eye_right = int(landmarks[133].x * image.shape[1])

    cloned_image = image.copy()
    cropped_left_eye = cloned_image[eye_top:eye_bottom, eye_left:eye_right]
    h, w, _ = cropped_left_eye.shape

    x = eye_left
    y = eye_top
    return x, y, w, h
