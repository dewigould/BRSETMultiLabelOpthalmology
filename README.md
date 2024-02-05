# BRESTMultiLabelOpthalmology
This repository hosts the version of the code used for the publication: "Multi-label disease classification of retinal fundus images using deep learning".


## Dataset Access
The dataset used in this project is publicly available on PhysioNet at https://physionet.org/content/brazilian-ophthalmological/1.0.0/. To obtain access, the user must apply to become a credentialed user and complete several AI ethics/ safety modules.

## Dependencies
To use this branch, you can run the following lines of code:
```
conda create -n BRESTMultiLabelOpth python==3.9
conda activate BRESTMultiLabelOpth
git clone https://github.com/dewigould/BRESTMultiLabelOpthalmology.git
cd BRESTMultiLabelOpth
pip install -e
```

## Getting Started
There are two scripts which can be run to train/ test models. The first is ``` MultiLabelClassifier/run.py''' and the second is ``` MultiLabelClassifier/run_ensemble.py'''. Each script contains a ```CONFIG''' dictionary where parameters can be chosen. These are
```
CONFIG = {
    num_epochs: \text{number of epochs},
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
```


To run baseline model:
```python
python MultiLabelClassifier/run.py
```

To run ensemble method:
```python
python MultiLabelClassifier/run_ensemble.py
```


## Citation
If you found our work useful, please consider citing:
