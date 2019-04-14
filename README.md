# Semantic Segmentation
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[image1]: ./Overview.jpg "Overview"
[image2]: ./NN_architecture.jpg "NN architecture"
[image3]: ./test_output/test_1.png "output image 1"
[image4]: ./test_output/test_4.png "output image 5"

## Introduction
In this project, a fully convolutional neural network (FCNN) is developed and trained in tensorflow to label the drivable area of a road images.

![alt_text][image1]

The project has been built according to this [rubric](https://review.udacity.com/#!/rubrics/989/view).

## Dependencies

* Python3 
* Tensorflow
* NumPy
* tqdm
* SciPy

## Usage 

### Training

1) Ensure the training data set is located in the `data\` directory. If not, run the `data/download_data.sh` script to download and extract the training data.
2) Run the following command to build, train and save the FCNN"
```
python train.py
```
In case you want to change the deep learniing archicture or the training options, you can do so from `train.py`.

### Inference

1) Ensure you have a trained deep learning model saved in the `model\` directory. If not, run the `model/download_model.sh` script to download and extract a pre-trained model.
2) Run the following command to run the pipeline on any png image of your choice:
```
python run.py input_image.png output_directory/
```
The python script will run the pipeline and generate labeled copies of the input images in the output_directory.

## Deep Learning Architecture & Training

### Architecture
The deep learning model implemented in this project utilizes a fully convolutional structure to label the drivable area. The Encoder part of the model is adopted from the `vgg16` structure while the decoder part was built from scratch.

The overall architecture of the final neural network model can be seen below:

![alt_text][image2]

### Training options
```
Optimizer: Adam optimizer
Learning rate: 0.00005
Btach size: 5
Epochs: 15
```

### Training Data
For training purposes, the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) was used. The dataset can be downloaded from [here](http://www.cvlibs.net/download.php?file=data_road.zip).

## Testing

The results of running the pipeline on the test set of the Kitti Road dataset can be seen in the [runs/](runs/) directory.

Additional images can be seen in the `test_images\` with their respective output in `test_output\`. Sample output images can be seen below:

Sample 1:

![alt_text][image3]

Sample 2:

![alt_text][image4]