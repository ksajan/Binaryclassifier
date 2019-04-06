# Binaryclassifier
  In this project i have implemented binary classifier for classification of human or not

<b>Requirements<b>
  1. Python 3
  2. Tensorflow
  3. Numpy
  4. Tqdm  - Instead giving interval it provide status bar
  
This is an approach to distiguish between Human and Dogs described [here](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/).
To run this code, First download the train dataset from this [link](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data).
Now use run the label.py with the train datset and remove of your choice labelled animal either Cat or Dog images. Now we are going to collect a dataset of images and label them for our train and test purpose to make a classification of Human or Not_Human. I use the data set [from](https://github.com/NVlabs/ffhq-dataset) as data set is huge due to high resolution. this dataset is been extracted using flicker api.

## DataSet
Note: I used Human faces as human and cat images as not_human for this binary classificxation between human or not. we can use other images( mixture of many images categories) but for simplcity I have used one type of category against human face dataset for this project.

Full dataset you can find [here](https://drive.google.com/drive/folders/1kDYxzDoSnUIk5tm2LEY1YFTBEJwpwvB3?usp=sharing) under mit lincence.
## Data Labeling
Now, Use label.py to make train and test dataset containg mixture of face and cats( In my case ) containing labeled and unlabeled images respectively. Train dataset containig 25000 images both cats and face of 12500 and test of 12500 image containig 6250 images of both.

## Model
### Prepocessing
We have got the data but first we need all of the images of same size, and then we can possibly gray scale them. Now, our first job is to convert label and images into numpy array so that we can pass through our network. Our images labeled like Not_human.1 or human.1 and so on, so we can spli out the Not_human/Human, and then we can convert them to array.

After saving ourdata as data array (data.npy) for easy use so thawe don't have to use again and again shape size, etc. 

### Defining Our Network

We will define 5 Layered Convulution neural network with a fully connected layer, and then the output layer. 
Now, it won't always be the same case that you're training the network fresh everytime. First we just want to see how 5 epochs trains, then, after the epochs are done, we can check for any number of epochs we wish to see. We need to save our model after every session and reloading it if we have a saved version. So we will add this.
Now, training data and testing data are both labeled datasets.The training data is what we will fit the neural network with and the test data is what we are going to use to validate the results. The test data will be "out of sample" meaning the testi8ng data will only be used to test the accuracyof the network not to train it. We have the test images that we have downloaded, which are not labeled. 
And finally we have distinguished between "human" and "not human", by plotting result usinmg matplotlib.


To run label.py:
```python
python label.py
```

There are 3 codes in label.py all three of them have different task according to our use use can remove comment in any of these use these can be use to label remove particular images or copy files from one director to another it uses OpenCv.

PS: If you find issue in this code please raise a issue or if you have update please do pull request. this Readme is under continuous update please follow to see the updates thanks
