# Add a new model to the DJL model zoo

This document outlines the procedure to add new models into the DJL model zoo.

## Step 1: Prepare the model files

The model files you will have or need depend on what type of model you have:

- Model built and trained in DJL:
  - <MODEL_NAME>-0000.params
- Model import from Apache MXNet:
  - <MODEL_NAME>-symbol.json
  - <MODEL_NAME>-0000.params

**Note:** To save space, compress the parameter file to .gz using the following command:

```shell
$ gzip <MODEL_NAME>-0000.params
```

You may also need to provide other artifact files for your model.
For example, a classification model requires
a `synset.txt` file to provide the names of the classes to classify into.
For an NLP model, you may need a `vocabulary.json` file to tokenize/untokenize the sentence.

## Step 2: Prepare the folder structure

1. Navigate to the `test/resources/mlrepo/model` folder and create a folder in it to store your model based on its category.
For example, `image_classification/ai/djl/resnet`.
2. Create a version folder within your newly created model's folder (e.g `0.0.1`). The version should match your ModelLoader class's version.
3. Copy model files into the version folder.

### Step 3: Create a `metadata.json` file

You need to create a `metadata.json` file for the model zoo to load the model. You can refer to the format in the `metadata.json` files for existing models to create your own.

For a model built as a DJL block, you must recreate the block before loading the parameters. As part of your `metadata.json` file, you should use the `arguments` property to specify the arguments required for the model loader to create another `Block` matching the one used to train the model.

**Note:** You need to update the sha1 hash of each file in your `metadata.json` file. Use the following command to get the sha1Hash value:

```shell
$ shasum -a 1 <file_name>
```

### Step 4: Upload your model

Verify that your folder has the following files (see Step 1 for additional files)

- <version>/<model_name>-00xx.params.gz
- metadata.json
- ...

The official DJL ML repository is located on an S3 bucket managed by the AWS DJL team.

For non-team members, coordinate with a team member in your pull request to add the necessary files.

For AWS team members, run the following command to upload your model to the S3 bucket:

```shell
$ ./gradlew syncS3
```

### Step 5: Open a PR to add your ModelLoader and metadata files to the git repository

You need to register your new model in the main model zoo interface. Be sure to include all the necessary information to load and use your model.

**Note**: Avoid checking in binary files to git. Binary files should only be uploaded to the S3 bucket.

### Step 6: Update the README file

Update the model zoo's README.md file for the appropriate model zoo to keep the list of models up to date.

