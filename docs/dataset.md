# Dataset

A dataset (or data set) is a collection of data that are used for machine-learning training job.

Machine learning typically works with three datasets:

- Training dataset

    The actual dataset that we use to train the model. The model learns weights and parameters from this data.
    
- Validation dataset

    The validation set is used to evaluate a given model during the training process. It helps machine learning
    engineers to fine-tune the [HyperParameter](https://github.com/awslabs/djl/blob/master/api/src/main/java/ai/djl/training/hyperparameter/param/Hyperparameter.java)
    at model development stage.
    The model doesn't learn from validation dataset; and validation dataset is optional.
    
- Test dataset

    The Test dataset provides the gold standard used to evaluate the model.
    It is only used once a model is completely trained.
 
See [Jason Brownlee’s article](https://machinelearningmastery.com/difference-test-validation-datasets/) for more detail.
 
## [Basic Dataset](../basicdataset/README.md)
 
DJL provides a number of built-in basic and standard datasets. These datasets are used to train deep learning models.
