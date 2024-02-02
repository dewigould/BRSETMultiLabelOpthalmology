
import numpy as np
import pandas as pd
import os, random, copy

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_curve, confusion_matrix, average_precision_score
from skimage.transform import rotate, AffineTransform, warp, resize

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, History
from tensorflow.keras.regularizers import l2

import MetricsFunctions as mf
import AuxilliaryFunctions as af
import DataGenerator as dg
import RetinalResNetModel as rrn


CONFIG = {
    num_epochs:6,
    input_batch_size=8,
    learning_rate = 0.00001,
    target_width=1000,
    path_to_dataset = "",
    input_test_name = "THE_NAME_OF_THE_EXPERIMENT",
    checkpoint_folder_name = "training_checkpoints/",
    num_normals_removed = 5000, #or NaN
    cols_to_be_amalgamated_into_other = ['retinal_detachment','hemorrhage','nevus','hypertensive_retinopathy'],
    train_on_one_camera_type = "Nikon", #Nikon, Canon, NaN
    use_weighted_loss_function = False,
    L2_penalty = 0, #Set to value required
    sampling_dictionary = {'nevus':500,'amd':1000} #disease and amount of over/undersampling
    }

checkpoint_path = CONFIG['checkpoint_folder_name']+CONFIG['input_test_name']+".h5"
path_to_pretrained_weights = "INSERT_PATH" + checkpoint_path
NUM_CLASSES=13

# Read in dataset
df=pd.read_csv(CONFIG['path_to_dataset'] + "labels.csv")

# Random under-sampling of normals col
if not np.isnan(CONFIG['num_normals_removed']):
    df = af.remove_random_subset_of_normal_entries(df,CONFIG['num_normals_removed'])

labels = df.columns[20:33]
class_names = list(labels)
labels_df = df[['image_id','diabetic_retinopathy', 'macular_edema', 'scar', 'nevus', 'amd', 'vascular_occlusion', 'hypertensive_retinopathy', 'drusens', 'hemorrhage', 'retinal_detachment', 'myopic_fundus', 'increased_cup_disc', 'other']]


# Combining columns into "Other"
if len(CONFIG['cols_to_be_amalgamated_into_other'])!=0:
    labels_df = af.consolidate_cols_into_other(labels_df,CONFIG['cols_to_be_amalgamanted_into_other'])
    NUM_CLASSES = 13-len(CONFIG['cols_to_be_amalgamanted_into_other'])
    for name in CONFIG['cols_to_be_amalgamanted_into_other']:
        class_names.remove(name)


# Generate train/ val/ test
train_inds, val_inds = train_test_split(np.array(list(range(labels_df.shape[0]))),test_size=0.2,random_state=2)
train_inds, test_inds = train_test_split(np.array(list(train_inds)),test_size=0.25,random_state=2)

train_df = labels_df.iloc[train_inds,:].reset_index(drop=True)
test_df = labels_df.iloc[test_inds,:].reset_index(drop=True)
val_df = labels_df.iloc[val_inds,:].reset_index(drop=True)

# Sensitivity Analysis
if not np.isnan(CONFIG['train_on_one_camera_type']):
    train_df, val_df, test_df = af.split_dataset_by_camera_type(df, CONFIG['train_on_one_camera_type'])

# Under/ Over-sampling.
for disease in CONFIG['sampling_dictionary'].keys():
    df_disease - train_df[train_df.disease==1]
    df_sample = df_disease.sample(CONFIG['sampling_dictionary'][disease],replace=True,random_state=42)
    train_df = pd.concat([train_df,df_sample]).reset_index(drop=True)


train_datagen = dg.DataGeneratorKeras(train = True, test=False,augmentation = True, preprocessing_fn = preprocess_input, batch_size = CONFIG['input_batch_size'],width=CONFIG['target_width'],height=CONFIG['target_width'],weighted_loss_function=CONFIG['use_weighted_loss_function'])
valid_datagen = dg.DataGeneratorKeras(train = False,test = False, augmentation = False, preprocessing_fn = preprocess_input, batch_size = CONFIG['input_batch_size'],width=CONFIG['target_width'],height=CONFIG['target_width'],weighted_loss_function=CONFIG['use_weighted_loss_function'])
test_datagen = dg.DataGeneratorKeras(train = False,test=True, augmentation = False, preprocessing_fn = preprocess_input, batch_size = 1,width=CONFIG['target_width'],height=CONFIG['target_width'],weighted_loss_function=CONFIG['use_weighted_loss_function'])

# Model
model = rrn.RetinalResNetModel(NUM_CLASSES,CONFIG['learning_rate'],height=CONFIG['target_width'],width=CONFIG['target_width'],l2_penalty = CONFIG['L2_penalty']).get_model()

# Model Checkpoints
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=False, mode='min')
history_checkpoints = History()
callbacks_list = [checkpoint,history_checkpoints]

# Model Fit
model_instance = rrn.RetinalResNetModel()
model_instance.load_model(model)
history_file_path = 'training_history_'+CONFIG['input_test_name']+'.pkl'
model_instance.do_fitting(training_data=train_datagen,validation_data=val_datagen,num_epochs = CONFIG['num_epochs'],call_backs=callbacks_list,history_file_path=history_file_path)

# Inference
model_instance = rrn.RetinalResNetModel()
model = load_model(path_to_pretrained_weights)
model_instance.load_model(model)

test_true, test_preds = af.get_truths_and_preds(num_classes=NUM_CLASSES,model=model,test_data = test_datagen)

# Do dynamic search for optimal threshold values.
optimal_thresholds = mf.get_optimal_thresholds(test_true,test_preds,class_names)
metric_df = mf.get_performance_metrics(test_true,test_preds,class_names,thresholds=optimal_thresholds)
metric_df.to_csv('performance_metrics_optimal_'+CONFIG['input_test_name']+'.csv')
