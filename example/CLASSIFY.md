Basic Classify Example
==============

In this tutorial, we will walk through the Classify model trained by MXNet.
Users can provide a image and do image classification with different models.


## Setup Guide

The available arguments are as follows:

| Argument   | Comments                                 |
| ---------- | ---------------------------------------- |
| `-c`       | Number of iterations in each test. |
| `-d`       | Duration of the test. |
| `-i`       | Image file. |
| `-l`       | Directory for output logs. |
| `-n`       | Model name. |
| `-p`       | Path to the model directory. |
| `-u`       | URL to download model archive. |

The available models are as follows:

| Models     |  
| ----------------  |
| caffenet          |
| nin               |
| Inception-BN       |
| inception_v1         |
| resnet-18 |
| squeezenet_v1.1   |
| squeezenet_v1.2   |
| vgg16             |
| vgg19             |

### Command line

```
cd example
./gradlew -Dmain=software.amazon.ai.example.ClassifyExample run --args="-n squeezenet_v1.1 -i ./src/test/resources/kitten.jpg"
```

```text
[INFO] Inference result: class: "tabby, tabby cat", probability: 0.7371954321861267
[INFO] inference P50: 66.185 ms, P90: 66.185 ms
```

### Intellij

1. Open `ClassifyExample.java`
2. Click `Edit Configuration` on the upper right side.  
![edit_config](../doc/img/editConfig.png)
3. In `Program arguments`, input the following arguments  
`-n squeezenet_v1.1 -i ./example/src/test/resources/kitten.jpg`  
![edit_config](../doc/img/programArguments.png)
4. Run `ClassifyExample.main()` 
![edit_config](../doc/img/run.png)

```text
[INFO] Inference result: class: "tabby, tabby cat", probability: 0.7371954321861267
[INFO] inference P50: 48.655 ms, P90: 48.655 ms
```
