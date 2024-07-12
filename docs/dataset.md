# Dataset

A dataset (or data set) is a collection of data that is used for training a machine learning model.

Machine learning typically works with three datasets:

- Training dataset

    The actual dataset that we use to train the model. The model learns weights and parameters from this data.
    
- Validation dataset

    The validation set is used to evaluate a given model during the training process. It helps machine learning
    engineers to fine-tune the [HyperParameters](https://github.com/deepjavalibrary/djl/blob/master/api/src/main/java/ai/djl/training/hyperparameter/param/Hyperparameter.java)
    at model development stage.
    The model doesn't learn from validation dataset; and validation dataset is optional.
    
- Test dataset

    The Test dataset provides the gold standard used to evaluate the model.
    It is only used once a model is completely trained.
    The test dataset should more accurately evaluate how the model will be performed on new data.
 
See [Jason Brownlee’s article](https://machinelearningmastery.com/difference-test-validation-datasets/) for more detail.
 
## [Basic Dataset](../basicdataset/README.md)
 
DJL provides a number of built-in basic and standard datasets. These datasets are used to train deep learning models.
This module contains the following datasets:

### CV

#### Image Classification

- [MNIST](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/cv/classification/Mnist.html) - A small and fast handwritten digits dataset
- [Fashion MNIST](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/cv/classification/FashionMnist.html) - A small and fast clothing type detection dataset
- [CIFAR10](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/cv/classification/Cifar10.html) - A dataset consisting of 60,000 32x32 color images in 10 classes
- [ImageNet](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/cv/classification/ImageNet.html) - An image database organized according to the WordNet hierarchy
  >**Note**: You have to manually download the ImageNet dataset due to licensing requirements.

#### Object Detection

- [Pikachu](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/cv/PikachuDetection.html) - 1000 Pikachu images of different angles and sizes created using an open source 3D Pikachu model
- [Banana Detection](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/cv/BananaDetection.html) - A testing single object detection dataset

#### Other CV

- [Captcha](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/cv/classification/CaptchaDataset.html) - A dataset for a grayscale 6-digit CAPTCHA task
- [Coco](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/cv/CocoDetection.html) - A large-scale object detection, segmentation, and captioning dataset that contains 1.5 million object instances
  - You have to manually add `com.twelvemonkeys.imageio:imageio-jpeg:3.11.0` dependency to your project

### NLP

#### Text Classification and Sentiment Analysis

- [AmazonReview](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/nlp/AmazonReview.html) - A sentiment analysis dataset of Amazon Reviews with their ratings
- [Stanford Movie Review](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/nlp/StanfordMovieReview.html) - A sentiment analysis dataset of movie reviews and sentiments sourced from IMDB
- [GoEmotions](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/nlp/GoEmotions.html) - A dataset classifying 50k curated reddit comments into either 27 emotion categories or neutral

#### Unlabeled Text

- [Penn Treebank Text](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/nlp/PennTreebankText.html) - The text (not POS tags) from the Penn Treebank, a collection of Wall Street Journal stories
- [WikiText2](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/nlp/WikiText2.html) - A collection of over 100 million tokens extracted from good and featured articles on wikipedia
 
#### Other NLP

- [Stanford Question Answering Dataset (SQuAD)](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/nlp/StanfordQuestionAnsweringDataset.html) - A reading comprehension dataset with text from wikipedia articles
- [Tatoeba English French Dataset](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/nlp/TatoebaEnglishFrenchDataset.html) - An english-french translation dataset from the Tatoeba Project

### Tabular

- [Airfoil Self-Noise](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/tabular/AirfoilRandomAccess.html) - A 6 feature dataset from NASA tests of airfoils
- [Ames House Pricing](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/tabular/AmesRandomAccess.html) - A 80 feature dataset to predict house prices
- [Movielens 100k](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/tabular/MovieLens100k.html) - A 6 feature dataset of movie ratings on 1682 movies from 943 users

### Time Series

- [Daily Delhi Climate](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/tabular/DailyDelhiClimate.html)