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
