# Object detection using a model zoo model

[Object detection](https://en.wikipedia.org/wiki/Object_detection) is a computer vision technique
for locating instances of objects in images or videos.

In this example we will use pre-trained model from [tensorflow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). 
The following code has been tested with EfficientDet, SSD MobileNet V2, Faster RCNN Inception Resnet V2,
but should work with most of tensorflow object detection models.

The source code can be found at [ObjectDetectionWithTensorflowSavedModel.java](https://github.com/deepjavalibrary/djl/blob/master/examples/src/main/java/ai/djl/examples/inference/ObjectDetectionWithTensorflowSavedModel.java).

## Setup guide

To configure your development environment, follow [setup](../../docs/development/setup.md).

## Run object detection example

### Pretrained SSD Model
 
The pre-trained SSD model can be found [here](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz).
You'll find a folder named ```ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model```. You need
to specify model name to let ModelZoo load the model from right location:

```java
Criteria<Image, DetectedObjects> criteria =  Criteria.builder()
    .setTypes(Image.class, DetectedObjects.class)
    .optModelUrls(modelUrl)
    // saved_model.pb file is in the subfolder of the model archive file
    .optModelName("ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model")
    ...
```

### Label mapping

The ms-coco labelmap can be downloaded from [here](https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt).
You can download this file and parse into labels in your ```Translator.prepare()``` class.

### Input image file
You can find the image used in this example in the project test resource folder: `src/test/resources/dog_bike_car.jpg`

![dogs](../src/test/resources/dog_bike_car.jpg)

### Build the project and run
Use the following command to run the project:

```sh
cd examples
./gradlew run -Dmain=ai.djl.examples.inference.ObjectDetectionWithTensorflowSavedModel
```

Your output should look like the following:

```text
[main] INFO - Detected objects image has been saved in: build\output\detected-dog_bike_car.png
[main] INFO - [
	{"class": "bicycle", "probability": 0.80220, "bounds": {"x"=0.147, "y"=0.209, "width"=0.576, "height"=0.603}}
	{"class": "car", "probability": 0.73779, "bounds": {"x"=0.596, "y"=0.145, "width"=0.297, "height"=0.149}}
	{"class": "dog", "probability": 0.72259, "bounds": {"x"=0.172, "y"=0.397, "width"=0.261, "height"=0.548}}
]
```

An output image with bounding box will be saved as build/output/detected-dog_bike_car.png:

![detected-dogs](img/detected-dog_bike_car.png)
