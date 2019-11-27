# FAQ

##### 1.  Why Deep Java Library (DJL)?

- Prioritizes the Java developer's experience
- Makes it easy for new machine learning developers to get started
- Allows developers to write modular, reusable code
- Reduces friction for deploying to a production environment
- Connects model developers with their consumers using the model zoo
- Allows developers to write code once and run it on any deep learning engine
- Allows developers to use engine specific features

##### 2. Which DL engines can I run with DJL?
While DJL is designed to be engine-agnostic and to run with the any engine, we currently
support only the MxNet engine. We have plans for supporting TensorFlow and PyTorch engines in the near future. 

##### 3. Does DJL support inference on GPU?
Yes. DJL does support inference on GPU. If GPUs are available, DJL automatically detects the GPU, and runs inference on a single GPU by default. 

##### 4. Does DJL support training on GPU?
Yes. DJL offers multi-GPU support. DJL can automatically detect if GPUs are available. If GPUs are available, it will
run on a single GPU by default, unless the user specifies otherwise.

During training, if you wish to train on multiple GPUs or if you wish to limit the number of GPUs to be used (you may want to limit the number of GPU for smaller datasets), you have to configure the `TrainingConfig` to do so by
setting the devices. For example, if you have 7 GPUs available, and you want the `Trainer` to train on 5 GPUs, you can configure it as follows. 

    int maxNumberOfGpus = 5;
    TrainingConfig config = new DefaultTrainingConfig(initializer, loss)
            .setOptimizer(optimizer)
            .addTrainingMetric(accuracy)
            .setBatchSize(batchSize)
            // Set the devices to run on multi-GPU
            .setDevices(Device.getDevices(numberOfGpus));
All of the examples in the example folder can be run on 
multiple GPUs with the appropriate arguments. Follow the steps in the example to [train a ResNet50 model on CIFAR-10 dataset](https://github.com/awslabs/djl/blob/master/examples/docs/train_cifar10_resnet.md#train-using-multiple-gpus) on a GPU.

##### 5. Does DJL support inference on multiple threads?
Yes. DJL offers multi-threaded inference. If using the MXNet engine for a multi-threaded inference case, you need to 
specify the 'MXNET_ENGINE_TYPE' environment variable to 'NaiveEngine'. For more information, see the
[Multi-threaded inference example](https://github.com/awslabs/djl/blob/master/examples/docs/multithread_inference.md).

##### 6. Can I run DJL on other versions of MxNet? 
This is not officially supported by DJL, but you can follow the steps outlined in the [troubleshooting document](https://github.com/awslabs/djl/blob/master/docs/development/troubleshooting.md#3-how-to-run-djl-using-other-versions-of-mxnet)
to use other versions of MXNet or built your own customized version.

##### 7. I have a model trained and saved by another DL engine. Can I load that model on to DJL?
While DJL is designed to be engine-agnostic, this feature is not officially supported by DJL yet. We are currently working to bring this feature to DJL soon. 


