# Ev-Eye: Event-based Eye Tracking with Motion Consistency

AIS 2024  Event-based Eye Tracking (CVPR Workshop 2024) 2nd Solution.

## Environment
- Ubuntu22.04
- Python3.9
- Nvidia GPU1080Ti

Using the following command line for the installation of dependencies rquired to trn Ev-Eye.
```
git clone https://github.com/MrFisher97/Solution_For_Event_based_Eye_Tracking_Challenge.git
cd Solution_For_Event_based_Eye_Tracking_Challenge
conda create -n EvEye python==3.9
conda activate EvEye
conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

#### Pretrained Model
- Google Drive: https://drive.google.com/drive/folders/1aDEr-jOR1JeqiV-N49DuED27sSKuCKwu?usp=drive_link 

We test the model from checkpooint `model_best_ep769_val_loss_0.0388.pth`.
The result are shown below:

|     |Eval|Test |
| --- | ---    | --- |
| Loss|0.0388|-|
|P5 acc|0.918|-|
|P10 acc|0.976|0.9948/0.9958|
|P15 acc|0.985|-|
|Dist|2.50|-|

## Instruction
#### Configuration
The configuration for training, evaluation and test is set on the `configs/task.yaml`, modify it according to your need.

#### Training
- You should change the `data_dir` setting in configs/taks.yaml to your dataset path, then execute the following command to start training:
```
python train.py --config_file task.yaml 
```

- This process will generate the `metadata` and `cached_dataset` directories which enable the fast data loading in training.
- The training script has already contain the testing phase using the model trained under all epoches.

#### Testing
- Running `test.py` scripy will get the tracking prediction on test split and store the result on `submission.csv` file
```
python test.py --config_file task.yaml --checkpoint your_model_path --log_dir store_result_path
```

#### Viualization
Using the following command to visualze the tracking result on eval/test split:
```
python visualize --config_file task.yaml --checkpoint your_model_path --split test
```

## Other details

## Acknowledgement
This code is based on the https://github.com/EETChallenge/challenge_demo_code