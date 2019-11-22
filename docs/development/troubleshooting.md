# Troubleshooting
Here we list some common problems you may face when using or developing DJL.
Please go through this list before submitting an issue.

## 1. IntelliJ throws the `No deep learning engine found` exception.
The following exception could appear after running `./gradlew clean`: 
```
16:57:55.313 [main] ERROR ai.djl.examples.training.util.AbstractTraining - Unexpected error
ai.djl.engine.EngineException: No deep learning engine found in class path.
	at ai.djl.engine.Engine.getInstance(Engine.java:81) ~[main/:?]
	at ai.djl.examples.training.util.Arguments.<init>(Arguments.java:42) ~[main/:?]
	at ai.djl.examples.training.util.AbstractTraining.runExample(AbstractTraining.java:67) [main/:?]
	at ai.djl.examples.training.TrainPikachu.main(TrainPikachu.java:72) [main/:?]
```
This issue is caused by IntelliJ and gradle runner mismatch. You can navigate to: `Preferences-> Build Execution Deployment -> Build Tools -> Gradle`,
 change `Build and running using:` to `Gradle`

If you prefer continue using `IntelliJ IDEA` as your runner, you can navigate to MXNet engine resources folder using the project view.  
```
mxnet -> mxnet-engine -> src -> main -> resources
```

Right click on resources folder and select `Rebuild<default>`.

![FAQ1](https://djl-ai.s3.amazonaws.com/web-data/images/FAQ_engine_not_found.png)


## 2. IntelliJ throws the `No Log4j 2 configuration file found.` exception.
The following exception could appear after running `./gradlew clean`: 
```bash
ERROR StatusLogger No Log4j 2 configuration file found. Using default configuration (logging only errors to the console), or user programmatically provided configurations. Set system property 'log4j2.debug' to show Log4j 2 internal initialization logging. See https://logging.apache.org/log4j/2.x/manual/configuration.html for instructions on how to configure Log4j 2
```
This is the same root cause in question above, you can follow the steps above to change `Build and running using:` to `Gradle`.
If you prefer continue using `IntelliJ IDEA` as your runner, navigate to the package of the program you are running using the project view.  
For example, if you are running a DJL example, navigate to:
```
examples -> src -> main -> resources -> log4j2.xml
```
Right click on `log4j2.xml` file and select `Recompile log4j2.xml`.

![FAQ2](https://djl-ai.s3.amazonaws.com/web-data/images/FAQ_log_recompile.png)

## 3. How to run DJL using other versions of MXNet?
**Note:** this is not officially supported by DJL, some functions may not work. 
If you really need some features in MXNet not provided by DJL, please submit an [issue](https://github.com/awslabs/djl/issues). 

By default DJL is running on the [MXNet engine](https://github.com/awslabs/djl/tree/master/mxnet/mxnet-engine),
 we use `mxnet-mkl` on CPU machines and `mxnet-cu101mkl` on GPU machines.
`mkl` means [Intel-MKLDNN](https://github.com/intel/mkl-dnn) is enabled.
`cu101` means [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) version 10.1 is enabled.

You don't need to download and install MXNet separately. It's automatically done when you
build the DJL project by running `./gradlew build`. However, you still have the option to use other versions of MXNet or built your own customized version. 

Follow the [MXNet Installation guide](https://mxnet.apache.org/get_started/?version=master&platform=linux&language=python&environ=pip&processor=cpu#) on how to install other versions of MXNet.
You need the latest MXNet to work with DJL, so remember to add `--pre` at the end of your `pip install` command.
After you installed MXNet, you just need to update the `MXNET_LIBRARY_PATH` environment variable with your `libmxnet.so` file location.
 
For example, if you are using an older version of CUDA(version 9.2), you can install MXNet with CUDA 9.2 by running the following command:
```bash
pip install mxnet-cu92 --pre
```
After installation, you can find the file location using the following commands in python:
```python
python
>>> import mxnet as mx
>>> mx.__file__
'//anaconda3/lib/python3.7/site-packages/mxnet/__init__.py'
```
Then update the environment variable value:
```bash
export MXNET_LIBRARY_PATH=//anaconda3/lib/python3.7/site-packages/mxnet/
```
Now you are all set, DJL will automatically use MXNet library from this location.