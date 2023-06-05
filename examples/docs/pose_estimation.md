# Pose estimation example

Pose estimation is a computer vision technique for determining the pose of an object in an image.

In this example, you learn how to implement inference code with a [ModelZoo model](../../docs/model-zoo.md) to detect people and their joints in an image.

The source code can be found at [PoseEstimation.java](https://github.com/deepjavalibrary/djl/blob/master/examples/src/main/java/ai/djl/examples/inference/PoseEstimation.java).

## Setup guide

Follow [setup](../../docs/development/setup.md) to configure your development environment.

## Run pose estimation example

### Input image file
You can find the image used in this example in the project test resource folder: `src/test/resources/pose_soccer.jpg`

![soccer](../src/test/resources/pose_soccer.png)

### Build the project and run
Use the following command to run the project:

```sh
cd examples
./gradlew run -Dmain=ai.djl.examples.inference.PoseEstimation
```

Your output should look like the following:

```text
[INFO ] - Pose image has been saved in: build/output/joints-0.png
[INFO ] - Pose image has been saved in: build/output/joints-1.png
[INFO ] - Pose image has been saved in: build/output/joints-2.png
[INFO ] - [
[
	{"Joint": {"x"=0.333, "y"=0.063}, "confidence": 0.6940},
	{"Joint": {"x"=0.333, "y"=0.031}, "confidence": 0.7182},
	{"Joint": {"x"=0.354, "y"=0.047}, "confidence": 0.4949},
	{"Joint": {"x"=0.354, "y"=0.047}, "confidence": 0.9011},
	{"Joint": {"x"=0.458, "y"=0.031}, "confidence": 0.8790},
	{"Joint": {"x"=0.375, "y"=0.172}, "confidence": 0.8546},
	{"Joint": {"x"=0.542, "y"=0.156}, "confidence": 0.8659},
	{"Joint": {"x"=0.417, "y"=0.313}, "confidence": 0.7731},
	{"Joint": {"x"=0.625, "y"=0.328}, "confidence": 0.9211},
	{"Joint": {"x"=0.458, "y"=0.500}, "confidence": 0.7541},
	{"Joint": {"x"=0.542, "y"=0.359}, "confidence": 0.5837},
	{"Joint": {"x"=0.458, "y"=0.469}, "confidence": 0.6387},
	{"Joint": {"x"=0.563, "y"=0.469}, "confidence": 0.6686},
	{"Joint": {"x"=0.271, "y"=0.703}, "confidence": 0.8583},
	{"Joint": {"x"=0.625, "y"=0.719}, "confidence": 0.8233},
	{"Joint": {"x"=0.125, "y"=0.969}, "confidence": 0.7007},
	{"Joint": {"x"=0.958, "y"=0.844}, "confidence": 0.7480}
], 
[
	{"Joint": {"x"=0.354, "y"=0.125}, "confidence": 0.8993},
	{"Joint": {"x"=0.375, "y"=0.109}, "confidence": 0.9235},
	{"Joint": {"x"=0.354, "y"=0.109}, "confidence": 0.8176},
	{"Joint": {"x"=0.438, "y"=0.094}, "confidence": 0.9242},
	{"Joint": {"x"=0.458, "y"=0.094}, "confidence": 0.6368},
	{"Joint": {"x"=0.500, "y"=0.156}, "confidence": 0.8452},
	{"Joint": {"x"=0.688, "y"=0.156}, "confidence": 0.6121},
	{"Joint": {"x"=0.479, "y"=0.250}, "confidence": 0.9007},
	{"Joint": {"x"=0.854, "y"=0.234}, "confidence": 0.7352},
	{"Joint": {"x"=0.208, "y"=0.250}, "confidence": 0.7154},
	{"Joint": {"x"=0.958, "y"=0.313}, "confidence": 0.5030},
	{"Joint": {"x"=0.625, "y"=0.484}, "confidence": 0.6673},
	{"Joint": {"x"=0.500, "y"=0.500}, "confidence": 0.7583},
	{"Joint": {"x"=0.708, "y"=0.719}, "confidence": 0.7621},
	{"Joint": {"x"=0.271, "y"=0.641}, "confidence": 0.8008},
	{"Joint": {"x"=0.250, "y"=0.906}, "confidence": 0.8605}
], 
[
	{"Joint": {"x"=0.271, "y"=0.156}, "confidence": 0.8428},
	{"Joint": {"x"=0.292, "y"=0.141}, "confidence": 0.8469},
	{"Joint": {"x"=0.271, "y"=0.125}, "confidence": 0.8029},
	{"Joint": {"x"=0.333, "y"=0.141}, "confidence": 0.9200},
	{"Joint": {"x"=0.354, "y"=0.141}, "confidence": 0.4879},
	{"Joint": {"x"=0.542, "y"=0.250}, "confidence": 0.8573},
	{"Joint": {"x"=0.292, "y"=0.250}, "confidence": 0.8553},
	{"Joint": {"x"=0.771, "y"=0.359}, "confidence": 0.9046},
	{"Joint": {"x"=0.167, "y"=0.391}, "confidence": 0.6416},
	{"Joint": {"x"=0.854, "y"=0.469}, "confidence": 0.9166},
	{"Joint": {"x"=0.188, "y"=0.359}, "confidence": 0.6091},
	{"Joint": {"x"=0.458, "y"=0.563}, "confidence": 0.5665},
	{"Joint": {"x"=0.375, "y"=0.563}, "confidence": 0.5728},
	{"Joint": {"x"=0.146, "y"=0.750}, "confidence": 0.6888},
	{"Joint": {"x"=0.667, "y"=0.766}, "confidence": 0.7807},
	{"Joint": {"x"=0.000, "y"=0.938}, "confidence": 0.2272},
	{"Joint": {"x"=0.396, "y"=0.828}, "confidence": 0.4885}
]]
```

Output images with the detected joints for each person will be saved in the build/output directory:

![joints-0](https://resources.djl.ai/images/joints-0.png)

![joints-1](https://resources.djl.ai/images/joints-1.png)

![joints-2](https://resources.djl.ai/images/joints-2.png)
