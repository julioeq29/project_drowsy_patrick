import cv2, os, time
from preprocessing import face_preprocess

def get_webcam_images():
    # Opens the inbuilt camera of laptop to capture video.
    cap = cv2.VideoCapture(0)
    print(cap)
    i = 0
    path = '../project_drowsy/data/images'

    while(cap.isOpened()):
        ret, frame = cap.read()
        # This condition prevents from infinite looping
        # incase video ends.
        if ret == False:
            break

        # Save Frame by Frame into disk using imwrite method
        cv2.imshow('Video', frame)
        # cv2.imwrite(os.path.join(path,'Frame'+str(i)+'.jpg'), frame)
        # i += 1
        #time.sleep(frequency)
        print("showing")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cap.release()
        cv2.destroyAllWindows()
