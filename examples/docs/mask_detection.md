# Mask detection with YOLOv5 - training and inference

YOLOv5 is a powerful model for object detection tasks. With the transfer learning technique, a pre-trained YOLOv5 model can be utilized in various customized object detection tasks with relatively small dataset. 

In this example, we apply it on the [Face Mask Detection dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection?select=images). We first train the YOLOv5s model in Python, with the help of [ATLearn](), a python transfer learning toolkit.
Then, the model is saved as an ONNX model, which is then imported into DJL for inference. We apply it on the mask wearing detection task. The source code can be found at [MaskDetectionOnnx.java](https://github.com/deepjavalibrary/djl/blob/master/examples/src/main/java/ai/djl/examples/inference/MaskDetectionOnnx.java)

## The training part in ATLearn

We initially attempted to import a pretrained YOLOv5 into DJL, and fine-tune it with the [Face Mask Detection dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection?select=images), similar to [Train ResNet for Fruit Freshness Classficiation](./train_transfer_fresh_fruit.md). However, YOLOv5 can not be converted to a PyTorch traced model, due to its data-dependent execution flow (see this [discussion](https://discuss.pytorch.org/t/yolov5-convert-to-torchscript/150180)), whick blocks the idea of retraining a Yolov5 model in DJL. So the training part is entirely in python.

The retraining of YOLOv5 can be found in an example in ATLearn: `examples/docs/face_mask_detection.md`. In this example, the YOLOv5 layers near the input are frozen while those near the output are fine-tuned with the customized data. This follows the transfer learning idea.

In this example, the trained model is exported to ONNX file and is then also used for inference in python, which will serve as a benchmark.

## Setup guide

To configure your development environment, follow [setup](../../docs/development/setup.md).

## Run mask detection example

### Input image file
We use the following image as input:

![mask](https://resources.djl.ai/images/face_mask_detection/face_mask.png)

### Build the project and run
Use the following command to run the project:

```
cd examples
./gradlew run -Dmain=ai.djl.examples.inference.MaskDetection
```

Your output should look like the following:

```text
[INFO ] - Detected objects image has been saved in: build/output/face_mask_result.png
[INFO ] - {
	"w/o mask": 0.8998132944107056,
	"w/ mask": 0.8930246829986572,
	"w/ mask": 0.8708265423774719,
	...
}
```

An output image with bounding box will be saved as `build/output/detected-mask-wearing.png`:

![detected-result](https://resources.djl.ai/images/face_mask_detection/face_mask_result.png)
