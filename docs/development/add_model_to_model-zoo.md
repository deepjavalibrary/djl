# Add a new Model to MXNet model zoo

This document outlines the procedure to add new models into MXNet model zoo.

## Step 1: Prepare model files
A MXNet symbolic model usually contains the following files:
- <MODEL_NAME>-symbol.json
- <MODEL_NAME>-0000.params

**Note:** To save space, please compress the parameter file to .gz:
```shell script
$ gzip <MODEL_NAME>-0000.params
```

You may also need to provide other artifact files for your model. For example, a classification model requires
a `synset.txt` to translate classes:
- synset.txt

For a NLP model, you may need a `vocabulary.json` to tokenize/untokenize the sentence:
- vocabulary.json

## Step 2: Prepare folder structure

1. Navigate to `test/resources/repo/model` and create a folder to store your model based on its category.
For example, `image_classification/ai/djl/mxnet/resnet`.
2. Create a version folder in your model's folder (e.g `0.0.1`), the version should match your ModelLoader class's version.
3. Copy model files into version folder

### Step 3: Create a `metadata.json` file
You need to create this file for model-zoo to load the model. You can refer to the format in the 'metadata.json' files for existing models to make your own.

**Note:** You need to update sha1 hash of each files in your `metadata.json` file. to get the sha1Hash value:

```shell script
$ shasum -a 1 <file_name>
```

### Step 4: Upload your model

Double check that you have the files:

- <version>/<model_name>-symbol.json
- <version>/<model_name>-00xx.params.gz
- metadata.json
- ...

Run the following command to upload to the S3 bucket:
```shell script
$ ./gradlew syncS3
```

### Step 5: Checkin your ModelLoader and metadata files to git
You need to register this model in the model zoo. Make all necessary changes to load and use your model.

**Note"**: Avoid checking in binary files to git. Binary files should only be uploaded to S3 bucket.

### Step 6: Update README file
Please update README.md file to keep the list of models up to date.

