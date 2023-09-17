import os
import numpy as np
import tensorflow as tf
import albumentations as A

# Import segmentation utils  
from utils.seg_backbone import *
from utils.seg_losses import *
from utils.seg_model_utils import *
from utils.seg_image_module import *
from utils.seg_evaluation import *

from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

''' 
You can change according to your GPU settings 
'''
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2' 

mirrored_strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))

SEED=2023 # Seed for reproductivity 
np.random.seed(SEED)
tf.random.set_seed(SEED)

seg_image_dir = '' # Directory of input images 
seg_mask_1_dir = '' # Directory of input ground truth masks (Cumulus)
seg_mask_2_dir = '' # Directory of input ground truth masks (Altocumulus)
seg_mask_3_dir = '' # Directory of input ground truth masks (Cirrocumulus)

IMAGE_SIZE = 256
IMG_CHANNELS = 1
EPOCHNUM = 100
BATCH_SIZE = 16
input_size = (IMAGE_SIZE, IMAGE_SIZE, IMG_CHANNELS)
result_dir = '' # Directory to save trained segmentation models 

MODEL = "attention_unet"

df = '' # Pandas dataframe which contains path to numpy, split, and etc. 

# Transformation to apply both to input image and ground truth mask
transform_both = A.Compose(
    [
        A.ElasticTransform(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1.0)
    ],
    additional_targets={'mask': 'image'}
)

# Transformation to apply only to input image 
transform_image_only = A.Compose(
    [
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(p=0.5),
        A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1.0)
    ])

# DENSITY 1 (Cumulus)

density = 1
train_generator_1 = DataGenerator(df, 1, "train", batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, transform_both=transform_both, transform_image_only=transform_image_only)
valid_generator_1 = DataGenerator(df, 1, "valid", batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, transform_both=transform_both, transform_image_only=transform_image_only)
test_generator_1 = DataGenerator(df, 1, "test", batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, transform_both=transform_both, transform_image_only=transform_image_only)

for i in range(1, 10):
    tl_alpha = 0.1 * i
    model = compile_model(learning_rate=3e-4, tl_alpha=tl_alpha, input_size=input_size)
    callback_list = callbacks(result_dir, f"Segmentation_density{density}_tlalpha0{i}")
    model.fit(train_generator_1, validation_data = valid_generator_1, epochs=EPOCHNUM, batch_size=BATCH_SIZE, callbacks=callback_list)
    K.clear_session()

# DENSITY 2 (Altocumulus)

density = 2
train_generator_2 = DataGenerator(df, 2, "train", batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, transform_both=transform_both, transform_image_only=transform_image_only)
valid_generator_2 = DataGenerator(df, 2, "valid", batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, transform_both=transform_both, transform_image_only=transform_image_only)
test_generator_2 = DataGenerator(df, 2, "test", batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, transform_both=transform_both, transform_image_only=transform_image_only)

for i in range(1, 10):
    tl_alpha = 0.1 * i
    model = compile_model(learning_rate=3e-4, tl_alpha=tl_alpha, input_size=input_size)
    callback_list = callbacks(result_dir, f"Segmentation_density{density}_tlalpha0{i}")
    model.fit(train_generator_2, validation_data = valid_generator_2, epochs=EPOCHNUM, batch_size=BATCH_SIZE, callbacks=callback_list)
    K.clear_session()
    
# DENSITY 3 (Cirrocumulus)

density = 3
train_generator_3 = DataGenerator(df, 3, "train", batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, transform_both=transform_both, transform_image_only=transform_image_only)
valid_generator_3 = DataGenerator(df, 3, "valid", batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, transform_both=transform_both, transform_image_only=transform_image_only)
test_generator_3 = DataGenerator(df, 3, "test", batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, transform_both=transform_both, transform_image_only=transform_image_only)

for i in range(1, 10):
    tl_alpha = 0.1 * i
    model = compile_model(learning_rate=3e-4, tl_alpha=tl_alpha, input_size=input_size)
    callback_list = callbacks(result_dir, f"Segmentation_density{density}_tlalpha0{i}")
    model.fit(train_generator_3, validation_data = valid_generator_3, epochs=EPOCHNUM, batch_size=BATCH_SIZE, callbacks=callback_list)
    K.clear_session()
