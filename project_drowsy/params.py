### GCP configuration - - - - - - - - - - - - - - - - - - -
BUCKET_NAME = 'wagon-data-753-rigg'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -
BUCKET_TRAIN_DATA_PATH = 'data/raw_data/train'

CLOUD_TRAIN_PATH = f'gs://{BUCKET_NAME}/project_drowsy/raw_data/train/'
TRAIN_PATH = 'raw_data/train/'
CLOUD_TEST_PATH = f'gs://{BUCKET_NAME}/project_drowsy/raw_data/test/'
TEST_PATH = 'raw_data/test/'

##### Model - - - - - - - - - - - - - - - - - - - - - - - -
# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'project_drowsy'
# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'V1'
