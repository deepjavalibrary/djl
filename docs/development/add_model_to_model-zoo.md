# Add a new Model to MXNet model zoo

This document outlines the procedure to add new models into the MXNet model zoo.

## Step 1: Prepare the model files
A MXNet symbolic model usually contains the following files:
- <MODEL_NAME>-symbol.json
- <MODEL_NAME>-0000.params

**Note:** To save space, compress the parameter file to .gz using the following command:
```shell script
$ gzip <MODEL_NAME>-0000.params
```

You may also need to provide other artifact files for your model. 

For example, a classification model requires
a `synset.txt` file to translate classes.

For a NLP model, you may need a `vocabulary.json` file to tokenize/untokenize the sentence.

## Step 2: Prepare the folder structure

1. Navigate to the `test/resources/repo/model` folder and create a folder in it to store your model based on its category.
For example, `image_classification/ai/djl/mxnet/resnet`.
2. Create a version folder within your newly created model's folder (e.g `0.0.1`). The version should match your ModelLoader class's version.
3. Copy model files into the version folder.

### Step 3: Create a `metadata.json` file
You need to create a `metadata.json` file for model-zoo to load the model. You can refer to the format in the 'metadata.json' files for existing models to create your own.

**Note:** You need to update the sha1 hash of each file in your `metadata.json` file. Use the following command to get the sha1Hash value:

```shell script
$ shasum -a 1 <file_name>
```

### Step 4: Upload your model

Verify that your folder has the following files:

- <version>/<model_name>-symbol.json
- <version>/<model_name>-00xx.params.gz
- metadata.json
- ...

Run the following command to upload your model to the S3 bucket:
```shell script
$ ./gradlew syncS3
```

### Step 5: Check in your ModelLoader and metadata files to git
You need to register your new model in the model zoo. Be sure to include all necessary information to load and use your model.

**Note**: Avoid checking in binary files to git. Binary files should only be uploaded to the S3 bucket.

### Step 6: Update the README file
Update the model-zoo/README.md file to keep the list of models up to date.

