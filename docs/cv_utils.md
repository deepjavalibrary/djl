# Computer Vision Utilities

DJL comes equipped with a number of helpful image processing and object detection utilities 
to make model creation and training as simple as possible.

## Using BufferedImageFactory to Read Images

The [BufferedImageFactory](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/modality/cv/BufferedImageFactory.html)
lets you create `Image`s from a variety of sources like URLs, local files, and input streams.

```java
// Load image from URL
URL url = new URL("https://s3.amazonaws.com/images.pdpics.com/preview/3033-bicycle-rider.jpg");
Image img = BufferedImageFactory.getInstance().fromUrl(url);
```

## Image Manipulation
[Image](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/modality/cv/Image.html) 
provides a suite of image manipulation functions to let you pre- and post-process your images all within DJL.

```java
int width = img.getWidth();
int height = img.getHeight();
Image leftHalfImg = img.getSubimage(0, 0, width / 2, height); // get left half of the image
NDManager manager = NDManager.newBaseManager();
NDArray imageArray = leftHalfImg.toNDArray(manager); // convert to NDArray
```

## Saving and Loading Your Images
Now that you have done your pre or post processing, you'll probably want to save your images for future use.
Simply call the `save` function from your `Image` and pass in an `OutputStream` and the image type (file extension).
Note: Some JDKs (openJDK) may not support saving JPG files when the image contains an alpha channel.

```java
// Save file
OutputStream out = new FileOutputStream("bicycle.png");
img.save(out, "png");
```
You can then use `BufferedImageFactory` to load it back in!

```java
// Load image from local file
Image imgLoaded = BufferedImageFactory.getInstance().fromFile(Path.of("bicycle.png"));
```

## Draw Bounding Boxes
`Image` includes a useful function to draw bounding boxes given a `DetectedObjects` instance
generated from a `ObjectDetection` model. We'll use the pre-trained `SingleShotDetection` model from the model zoo 
to demonstrate below.

```java
// Load Object Detection Model
Criteria<Image, DetectedObjects> criteria = Criteria.builder()
        .setTypes(Image.class, DetectedObjects.class)
        .optArtifactId("ssd")
        .build();
ZooModel<Image, DetectedObjects> model = criteria.loadModel();
Predictor<Image, DetectedObjects> predictor = model.newPredictor();

// Detect Objects
DetectedObjects detectedObjects = predictor.predict(img);

// Draw Bounding Boxes
img.drawBoundingBoxes(detectedObjects);

// Save Image with Bounding Boxes
OutputStream out1 = new FileOutputStream("bicycleBoundBox.png");
img.save(out1, "png");
```

## Draw Joints
You can also draw joints if you have a `Joints` instance generated from a `PoseEstimation` model.
Here, we'll use the `SimplePose` model from the model zoo!

```java
// Load Pose Detection Model
Criteria<Image, Joints> criteria = Criteria.builder()
        .setTypes(Image.class, Joints.class)
        .optArtifactId("simple_pose")
        .build();
Predictor<Image, Joints> predictor = model.newPredictor();

// Detect Joints
Joints joints = predictor.predict(img);

// Draw Joints
img.drawJoints(joints);

// Save Image with Joints
OutputStream out2 = new FileOutputStream("bicycleJoints.png");
img.save(out2, "png");
```

## Useful Information
If you want to learn more about loading models, click [here](http://docs.djl.ai/docs/load_model.html).

If you want to learn more about the model zoo, click [here](http://docs.djl.ai/docs/model-zoo.html).
