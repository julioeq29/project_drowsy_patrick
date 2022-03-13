import os
import random
from project_drowsy.params import *
from tensorflow.io.gfile import glob

def get_train_faces(**params):

    '''get the file names for the train faces as a dictionary'''

    local = params.get('local')
    n_images = params.get('n_images')
    random_seed = params.get('random_seed')

    if local:
        yawn_train_img = os.listdir(f"{TRAIN_PATH}yawn")
        no_yawn_train_img = os.listdir(f"{TRAIN_PATH}no_yawn")
    else:
        yawn_train_img = glob(f"{CLOUD_TRAIN_PATH}yawn/" + "*.jpg")
        no_yawn_train_img = glob(f"{CLOUD_TRAIN_PATH}no_yawn/"+"*.jpg")
        yawn_train_img = [i.split("/")[-1] for i in yawn_train_img]
        no_yawn_train_img = [i.split("/")[-1] for i in no_yawn_train_img]

    random.seed(random_seed)
    random.shuffle(yawn_train_img)
    random.shuffle(no_yawn_train_img)

    if n_images:
        slicing = slice(n_images)
        yawn_train_img = yawn_train_img[slicing]
        no_yawn_train_img = no_yawn_train_img[slicing]

    images_train_dict = {
        'yawn': yawn_train_img,
        'no_yawn': no_yawn_train_img
    }
    return images_train_dict

def get_test_faces(**params):
    '''get the file names for the test faces as a dictionary'''
    local = params.get('local')
    n_images = params.get('n_images')
    random_seed = params.get('random_seed')

    if local:
        yawn_test_img = os.listdir(f"{TEST_PATH}yawn")
        no_yawn_test_img = os.listdir(f"{TEST_PATH}no_yawn")
    else:
        yawn_test_img = glob(f"{CLOUD_TEST_PATH}yawn/" + "*.jpg")
        no_yawn_test_img = glob(f"{CLOUD_TEST_PATH}no_yawn/" + "*.jpg")

        yawn_test_img = [i.split("/")[-1] for i in yawn_test_img]
        no_yawn_test_img = [i.split("/")[-1] for i in no_yawn_test_img]

    random.seed(random_seed)
    random.shuffle(yawn_test_img)
    random.shuffle(no_yawn_test_img)

    if n_images:
        slicing = slice(n_images)
        yawn_test_img = no_yawn_test_img[slicing]
        no_yawn_test_img = no_yawn_test_img[slicing]

    images_test_dict = {
        'yawn': yawn_test_img,
        'no_yawn': no_yawn_test_img
    }

    return images_test_dict

def get_train_eyes(**params):
    '''get the file names for the train eyes as a dictionary'''
    local = params.get('local')
    n_images = params.get('n_images')
    random_seed = params.get('random_seed')

    if local:
        open_train_imgs = os.listdir(f"{TRAIN_PATH}Open")
        closed_train_imgs = os.listdir(f"{TRAIN_PATH}Closed")
    else:
        open_train_imgs = glob(f"{CLOUD_TRAIN_PATH}Open/" + "*.jpg")
        closed_train_imgs = glob(f"{CLOUD_TRAIN_PATH}Closed/" + "*.jpg")
        #grab filenames from full path
        open_train_imgs = [i.split("/")[-1] for i in open_train_imgs]
        closed_train_imgs = [i.split("/")[-1] for i in closed_train_imgs]

    random.seed(random_seed)
    random.shuffle(open_train_imgs)
    random.shuffle(closed_train_imgs)

    if n_images:
        slicing = slice(n_images)
        open_train_imgs = open_train_imgs[slicing]
        closed_train_imgs = closed_train_imgs[slicing]

    images_train_dict = {
        'Open': open_train_imgs,
        'Closed': closed_train_imgs
    }
    return images_train_dict

def get_test_eyes(**params):
    '''get the file names for the test eyes as a dictionary'''
    local = params.get('local')
    n_images = params.get('n_images')
    random_seed = params.get('random_seed')

    if local:
        open_test_imgs = os.listdir(f"{TEST_PATH}Open")
        closed_test_imgs = os.listdir(f"{TEST_PATH}Closed")
    else:
        open_test_imgs = glob(f"{CLOUD_TEST_PATH}Open/" + "*.jpg")
        closed_test_imgs = glob(f"{CLOUD_TEST_PATH}Closed/" + "*.jpg")

        open_test_imgs = [i.split("/")[-1] for i in open_test_imgs]
        closed_test_imgs = [i.split("/")[-1] for i in closed_test_imgs]

    random.seed(random_seed)
    random.shuffle(open_test_imgs)
    random.shuffle(closed_test_imgs)

    if n_images:
        slicing = slice(n_images)
        open_test_imgs = open_test_imgs[slicing]
        closed_test_imgs = closed_test_imgs[slicing]

    images_test_dict = {
        'Open': open_test_imgs,
        'Closed': closed_test_imgs
    }
    return images_test_dict

if __name__ == "__main__":
    test_params = dict(local=False,
                  random_seed=1)
    face_image_files = get_test_faces(**test_params)

    print(face_image_files)
