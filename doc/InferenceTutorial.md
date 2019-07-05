# Build your first Inference Application

## Introduction
Welcome to the Joule world.
Joule is an API designed to deal with all kinds of Deep Learning tasks.
Users will be able to create, train and do inference with Deep Learning models.

In this tutorial, we will guide you to create your first application to use Joule for Deep Learning Inference.
We will implement an Object Detection Application based on pre-trained Resnet50SSD model.

## Prerequisite
Before we start, please see the JavaDoc of the following classes.
These are the core component we are using to load the pre-trained model and do inference.

- [Model](https://joule.s3.amazonaws.com/java-api/software/amazon/ai/Model.html)
- [Predictor](https://joule.s3.amazonaws.com/java-api/software/amazon/ai/inference/Predictor.html)
- [Translator](https://joule.s3.amazonaws.com/java-api/software/amazon/ai/Translator.html)
- [NDArray](https://joule.s3.amazonaws.com/java-api/software/amazon/ai/ndarray/NDArray.html) and [NDList](https://joule.s3.amazonaws.com/java-api/software/amazon/ai/ndarray/NDList.html)

## Start Implementation
The workflow looks like the following:

![image](img/workFlow.png)


### Step 0 Include Dependencies

To include Joule in your project, add the following dependencies to your build.gradle file, or corresponding entry
pom.xml.

~~~
compile "software.amazon.ai:joule-api:0.1.0"
runtime "org.apache.mxnet:joule:0.1.0"
runtime "org.apache.mxnet:mxnet-native-mkl:1.5.0-SNAPSHOT:osx-x86_64"
~~~



### Step 1 Implement the Translator

You can start by implementing the `Translator` interface explained above. The `Translator` is the unit that 
converts user-defined input and output objects to NDList and vice versa. To this end, the Translator 
interface has two methods that need to be implemented: `processInput()` and `processOutput()`. This is 
represented as the two white boxes in the image. These are the two main blocks of code that the user needs to
implement. The rest of the blocks can be used as is to run inference. 

##### Pre-processing

The input to the inference API can be any user-defined object. The `processInput()` method in the user
implementation of `Translator` must convert the user-defined input object into an `NDList`. 

The SSD Example will take an image as a NCHW format:
- N: Batch size
- C: Channel
- H: Height
- W: Width

The input of the translator should be an buffered image or any other type that 
can load an image. To simplify our inference experience, the batch size will be 1. Channel is usually 3 (RGB).
For Height and Width, we recommend to use (512, 512) since the model was trained on images input.

##### Post-processing

The output of the inference API can also be any user-defined object. The `processOutput()` method in the user
implementation of `Translator` must convert `NDList` to the required object. 

The SSD Example will return a list of `DetectedObjects` as it's output.

### Step 2 Load the model

Loading a model requires the path to the directory where the model is stored, and the name of the model. 
~~~
Model model = Model.loadModel(modelDir, modelName);
~~~


### Step 3 Create Predictor

Once the model is loaded, we have everything we need to create a Predictor that can run inference. We have 
implemented a Translator, and loaded a model. We can create a Predictor using these objects. 

~~~
Predictor<BufferedImage, List<DetectedObject>> ssd = Predictor.newInstance(model, translator, context)
~~~

The Predictor class extends AutoCloseable. Therefore, it is good to use it within a try-with-resources block. 

### Run Inference

We can can use the Predictor create above to run inference in one single step!
~~~
List<DetectedObject> predictResult = predictor.predict(img);
~~~

The example provided in this module applies the bounding boxes to a copy of the original image, stores the result
a file called ssd.jpg in the provided output directory. 

The model, image, output directory can all be provided as input. The available arguments are as follows:
 
 | Argument   | Comments                                 |
 | ---------- | ---------------------------------------- |
 | `-c`       | Number of iterations in each test. |
 | `-d`       | Duration of the test. |
 | `-i`       | Image file. |
 | `-l`       | Directory for output logs. |
 | `-n`       | Model name. |
 | `-p`       | Path to the model directory. |
 | `-u`       | URL to download model archive. |
 
 
 You can get the model from running the following commands
 
 ~~~
 curl -O https://s3.amazonaws.com/model-server/model_archive_1.0/examples/ssd/resnet50_ssd_model-symbol.json
 curl -O https://s3.amazonaws.com/model-server/model_archive_1.0/examples/ssd/resnet50_ssd_model-0000.params
 ~~~
 
 You can navigate to the source folder, and simply type the following command to run the inference:
 
 ```
 ./gradlew -Dmain=software.amazon.ai.example.SsdExample run --args="-p build/ -n resnet50_ssd_model -i {PATH_TO_IMAGE} -l {OUTPUT_DIR}"
 ```
 

 





