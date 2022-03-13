# Import for streamlit
import streamlit as st
import av
import time
import simpleaudio as sa
from tensorflow.keras.models import load_model

from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

#Import for handling image
from cv2 import cv2
from PIL import Image
# Models and preprocessing
from project_drowsy.predict import make_prediction, mapping

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

#download model
@st.cache(allow_output_mutation=True)
def retrieve_model():
    face_model = load_model('project_drowsy/models/project_drowsy_models_face_model.h5')
    eye_model = load_model('project_drowsy/models/project_drowsy_models_eye_model.h5')
    return face_model, eye_model


#Main intelligence of the file, class to launch a webcam, detect faces, then detect drowsiness and output probability for drowsiness
def app_drowsiness_detection():
    class DrowsinessPredictor(VideoProcessorBase):

        def __init__(self) -> None:
            self.drowsy_counter = 0
            self.counter = 0
            self.drowsy_flag = False
            self.face_model = retrieve_model()[0]
            self.eye_model = retrieve_model()[1]

        def draw_and_predict(self, image):

            preprocess_params = dict(webcam=image,
                                    image_size=145,
                                    predict=True)

            try:
                print(self.counter)

                if self.counter % 10 == 0:

                    if self.drowsy_flag:
                        wave_obj = sa.WaveObject.from_wave_file('airhorn.wav')
                        play_obj = wave_obj.play()
                        play_obj.wait_done()
                        #time.sleep(1) # Sleep for 1 second
                        self.drowsy_flag=False



                    cropped_face, face_coords, cropped_left_eye, cropped_right_eye  = make_prediction(**preprocess_params)
                    # prediction, probs, face_coords, left_eye_coords, right_eye_coords  = make_prediction(**preprocess_params)

                    def to_predict(cropped_face, face_coords, cropped_left_eye, cropped_right_eye):
                      yawn_prob = self.face_model.predict(cropped_face)[0][0]
                      left_eye_prob = self.eye_model.predict(cropped_left_eye[0])[0][0]
                      right_eye_prob = self.eye_model.predict(cropped_right_eye[0])[0][0]
                      probs = [yawn_prob, left_eye_prob, right_eye_prob]

                      yawn_prediction = (yawn_prob > 0.5).astype("int32")
                      left_eye_prediction = (left_eye_prob > 0.5).astype("int32")
                      right_eye_prediction = (right_eye_prob > 0.5).astype("int32")
                      prediction = mapping(yawn_prediction, left_eye_prediction, right_eye_prediction)

                      return prediction, probs, face_coords, cropped_left_eye[1], cropped_right_eye[1]

                    prediction, probs, face_coords, left_eye_coords, right_eye_coords = to_predict(cropped_face, face_coords, cropped_left_eye, cropped_right_eye)

                    # draw eye bounding boxes using co-ordinates of the bounding box (from preprocessing)
                    xmin_l,xmax_l,ymin_l,ymax_l = left_eye_coords
                    xmin_r, xmax_r, ymin_r, ymax_r = right_eye_coords
                    #left eye
                    cv2.rectangle(image, (xmax_l,ymax_l), (xmin_l,ymin_l),
                                color=(0, 255, 0), thickness=2)
                    #right eye
                    cv2.rectangle(image, (xmax_r, ymax_r), (xmin_r, ymin_r),
                                color=(0, 255, 0),
                                thickness=2)

                    #draw face box
                    #print(face_coords)
                    xmin, xmax, ymin, ymax = face_coords
                    cv2.rectangle(image, (xmax, ymax), (xmin, ymin),
                                color=(0, 255, 0),
                                thickness=3)

                    # evaluate
                    print(prediction)
                    print(probs)
                    # # Put text on image


                    if ("closed" in prediction) or ("yawn" in prediction):
                        self.drowsy_counter += 1
                        if self.drowsy_counter >= 5:
                            self.drowsy_flag=True
                            text = "Prediction = Drowsy"
                            colour = (255, 0, 0)
                        else:
                            text = "Prediction = Alert"
                            colour = (0, 255, 0)
                        cv2.putText(image, text, (40, 40),
                                cv2.FONT_HERSHEY_PLAIN, 2, colour, 2)
                    else:
                        self.drowsy_counter = 0
                        text = "Prediction = Alert"
                        cv2.putText(image, text, (40, 40),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                # ####################

                self.counter += 1

            except Exception as e:
                print(e)
                text = 'WARNING! DRIVER NOT FOUND!'
                print(text)
                cv2.putText(image, text, (40, 40),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            return image


        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="rgb24")
            annotated_image = self.draw_and_predict(image)
            return av.VideoFrame.from_ndarray(annotated_image, format="rgb24")

    webrtc_ctx = webrtc_streamer(
        key="drowsiness-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=DrowsinessPredictor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

def about():
    st.write('Welcome to our drowsiness detection system')
    st.markdown("""

     **About our app**
    - We are attempting to reduce the prevalence of car accidents caused by driver drowsiness.

    - Our app detects from a live webcam whether a driver is drowsy and if so, alerts them.

    - We consider a driver as drowsy if they are yawning or they have their eyes closed.

    **Examples**

    """)


    alert_image = Image.open('alert_example.png')
    st.image(alert_image, caption='Alert driver')

    drowsy_image = Image.open('drowsy_example.png')
    st.image(drowsy_image, caption='Drowsy driver')


def pre_recorded():

    # # replace sample.mp4 for pre-recorded video
    # video_file = open("sample.mp4", "rb").read()
    # st.video(video_file, start_time = 3)

    st.video("https://www.youtube.com/watch?v=DrKLYvLPidw")



############################ Sidebar + launching #################################################


def main():
    object_detection_page = "Live Video Detector"
    pre_recorded_video = "Pre-recorded Video"
    about_page = "About Drowsiness Detection"

    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        [
            about_page,
            object_detection_page,
            pre_recorded_video
        ],
    )
    st.subheader(app_mode)

    if app_mode == about_page:
        about()

    if app_mode == object_detection_page:
        app_drowsiness_detection()

    if app_mode == pre_recorded_video:
        pre_recorded()


if __name__ == '__main__':
    main()
