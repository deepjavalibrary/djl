# MXNet model zoo

## Introduction

MXNet model zoo contains most of the Symbolic model that can be used for inference and training.

## Add new Model

Please follow the step to step instruction to add your new model

### Step 1: Create the folder

go to `test/resources/repo/model` and find the type of the model you are looking for.
Then create the corresponding folder to store your model. For example, `image_classification.org.apache.mxnet.resnet`.

### Step 2: Place your model files
Please create a folder with the version number (e.g `0.0.1`) and upload the model files.

- symbol
- params.gz

Note, please zip the params to save the space.

### Step 3: Place your model artifacts
You may need to provide the artifact files for your model.
It should be placed outside the version number folder.

For cv or classification model, you may need a `synset.txt` to translate classes.

For nlp model, you may need a `vocabulary.json` to tokenize/untokenize the sentence.

### Step 4: Create a `metadata.json`
You need to create this file for modelzoo to load the model.
You can refer the format from the existing ones and make your own.

run `shasum -a 1 <file_name>` to get the sha1Hash value.

### Step 5: Wrap it up
Now you need to register this model in the model zoo.
Please make all necessary changes to load and use your model.

### Step 6: Upload your model

Double check if you have the files

- <version>/<model_name>-symbol.json
- <version>/<model_name>-00xx.params.gz
- metadata.json
- ...

```shell script
$ ./gradlew syncS3
```

### Step 7:
Please avoid checking in binary files to git. Binary files should only be uploaded to S3 bucket.
Remove all the files except the metadata.json and checkin the code.
