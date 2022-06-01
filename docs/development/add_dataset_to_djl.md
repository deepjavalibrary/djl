# Add a new dataset to DJL basic datasets

This document outlines the procedure to add new datasets to DJL.

## Step 1: Prepare the folder structure

1. Navigate to the `test/resources/mlrepo/dataset` folder and create a folder in it to store your dataset based on its category.
   For example, `cv/ai/djl/basicdataset/mnist`.
2. Create a version folder within your newly created dataset's folder (e.g `0.0.1`). The version should match your dataset version.

### Step 2: Create a `metadata.json` file

You need to create a `metadata.json` file for the repository to load the dataset. You can refer to the format in the `metadata.json` files for existing datasets to create your own.

**Note:** You need to update the sha1 hash of each file in your `metadata.json` file. Use the following command to get the sha1Hash value:

```shell
$ shasum -a 1 <file_name>
```

### Step 3: Create a Dataset implementation

Create a class that implements the dataset and loads it.
For more details on creating datasets, see the [dataset creation guide](how_to_use_dataset.md).
You should also look at examples of official DJL datasets such as [`AmesRandomAccess`](https://github.com/deepjavalibrary/djl/blob/master/basicdataset/src/main/java/ai/djl/basicdataset/tabular/AmesRandomAccess.java)
or [`Cifar10`](https://github.com/deepjavalibrary/djl/blob/master/basicdataset/src/main/java/ai/djl/basicdataset/cv/classification/Cifar10.java).

Then, add some tests for the dataset.
For testing, you can use a local repository such as:

```java
Repository repository = Repository.newInstace("testRepository", Paths.get("/test/resources/mlrepo"));
```

### Step 4: Update the datasets list

Add your dataset to the [list of built-in datasets](../dataset.md).

### Step 5: Upload metadata

The official DJL ML repository is located on an S3 bucket managed by the AWS DJL team.
You have to add the metadata and any dataset files to the repository.

For non-AWS team members, go ahead straight to Step 6 and open a pull request.
Within the pull request, you can coordinate with an AWS member to add the necessary files.

For AWS team members, run the following command to upload your model to the S3 bucket:

```shell
$ ./gradlew syncS3
```

The `metadata.json` in DJL is mainly a repository of metadata.
Within the metadata is typically contains only links indicating where the actual data would be found.

However, some datasets can be distributed by DJL depending on whether it makes it easier to use and the dataset permits redistribution.
In that case, coordinate with an AWS team member in your pull request.

### Step 6: Open a PR to add your ModelLoader and metadata files to the git repository

**Note**: Avoid checking in binary files to git. Binary files should only be uploaded to the S3 bucket.

If you are relying on an AWS team member, you should leave your code with the local test repositories.
If you try to use the official repository before it contains your metadata, the tests will not pass the CI.

Once an AWS team member adds your metadata, they will prompt you to update your PR to the official repository.
