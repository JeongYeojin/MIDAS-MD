import os
import keras 
from albumentations import * 
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from seg_backbone import *
from seg_losses import *

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

def compile_model(learning_rate, tl_alpha, input_size):
    
    # Define model
    optimizer = Adam(learning_rate=learning_rate)
    lossfxn = ftl_wrapper(tl_alpha)
    final_loss = tl_wrapper(tl_alpha)

    loss = {'pred1':lossfxn, # focal tversky loss (FTL)
            'pred2':lossfxn, # focal tversky loss (FTL)
            'pred3':lossfxn, # focal tversky loss (FTL)
            'final': final_loss} # Tversky loss (TL) 

    loss_weights = {'pred1':1,
                    'pred2':1,
                    'pred3':1,
                    'final':1}
    
    ## Load improved attention u-net 
    # with mirrored_strategy.scope():
    model = attn_reg(input_size)
    model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=[dsc])
        
    return model

def load_model(learning_rate, tl_alpha, input_size, ckpt_path):
    # with mirrored_strategy.scope():
    model = compile_model(learning_rate, tl_alpha, input_size)
    model.load_weights(ckpt_path)
    return model

class MetricTracker(keras.callbacks.Callback):
    
    def __init__(self, result_dir, hist_name):
        self.HIST_NAME = hist_name
        self.result_dir = result_dir
        self.hist_df = pd.DataFrame()
        
    def on_epoch_end(self, epoch, logs):
        new_row = {key : value for (key, value) in logs.items()}
        self.hist_df = self.hist_df.append(new_row, ignore_index=True)
        self.hist_df.to_csv(os.path.join(self.result_dir, f"{self.HIST_NAME}.csv"), header=True, index=False)

def callbacks(result_dir, ckpt_name):
        
    ## Callbacks
    ckpt_path = os.path.join(result_dir, f"{ckpt_name}.ckpt")
    
    check_point = ModelCheckpoint(
        ckpt_path, 
        monitor='val_final_dsc', 
        verbose=1, 
        save_best_only=True, 
        save_weights_only=True, 
        mode='max')

    reduceLROnPlat = ReduceLROnPlateau(
          monitor="val_loss",
          factor=0.1,
          patience=5,
          verbose=1,
          mode="min",
          min_delta=0.0001,
          cooldown=2,
          min_lr=1e-6)
    
    early = EarlyStopping(monitor="val_final_dsc", mode="max", patience=10)

    metrictracker = MetricTracker(result_dir, ckpt_name)

    callbacks_list = [check_point, early, reduceLROnPlat, metrictracker]
    
    return callbacks_list
