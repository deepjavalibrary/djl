# Dataset Creation

The Dataset in DJL represents both the raw data and loading process.
RandomAccessDataset implements the Dataset interface and provides comprehensive data loading functionality.
RandomAccessDataset is also a basic dataset that supports random or sequential access of data using indices.
You can easily customize your own dataset by extending RandomAccessDataset.

We provide several well-known datasets that you can use in our [Basic Datasets module](https://javadoc.io/doc/ai.djl/basicdataset/latest/index.html).

We also provide several built-in datasets that you can easily wrap around existing NDArrays and images.

- ArrayDataset - a dataset that wraps your existing NDArrays as inputs and labels.
- ImageFolder - a dataset that wraps your images under folders for image classification.

If none of the provided datasets meet your requirements, you can also easily customize you own dataset by extending
the RandomAccessDataset.

## How to use the ArrayDataset

The following code illustrates an implementation of [ArrayDataset](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/training/dataset/ArrayDataset.html).
The ArrayDataset is recommended only if your dataset is small enough to fit in memory.

```java
// given you have data1, data2 and label1, label2, label3
ArrayDataset dataset = new ArrayDataset.Builder()
                              .setData(data1, data2)
                              .optLabels(label1, label2, label3)
                              .setSampling(20, false)
                              .build();

```

When you get the `Batch` from `trainer.iterateDataset(dataset)` or `dataset.getData(manager)`,
you can use ``batch.getData()`` to get a NDList. The size of this NDList is the amount of data NDArrays 
entered in `setData()`, in this case the size is 2. You can then use `NDList.get(0)` to get your first 
data and `NDList.get(1)` to get your second data.
Similarly, you can use `batch.getLabels()` to get a NDList with size 3.

## How to use the ImageFolder dataset

This section outlines the procedure to use the ImageFolder dataset.

The ImageFolder dataset is recommended only if you want to iterate through your images with their single label. For example, when training classic image classification.

### Step 1: Prepare Images
Arrange your image folder structure as follows:

```
dataset_root/shoes/Aerobic Shoes1.png
dataset_root/shoes/Aerobic Shose2.png
...
dataset_root/boots/Black Boots.png
dataset_root/boots/White Boots.png
...
dataset_root/pumps/Red Pumps
dataset_root/pumps/Pink Pumps
...
```

The dataset will take the folder name e.g. `boots`, `pumps`, `shoes` in sorted order as your labels
**Note:** Nested folder structures are not currently supported

### Step 2: Use the Dataset
Add the following code snippet to your project to use the ImageFolder dataset.

```java
ImageFolder dataset =
    new ImageFolder.Builder()
    // input the root string
    .setRepository(new SimpleRepository(Paths.get("root")))
    .optPipeline(
        // create preprocess pipeline you want
        new Pipeline()
            .add(new Resize(width, height))
            .add(new ToTensor()))
    .setSampling(batchSize, true)
    .build();
// call prepare before using
dataset.prepare();

// to get the synset
List<String> synset = dataset.getSynset();
```

Typically, you would add pre-processing pipelines like Resize(), in order to batchify the dataset, and ToTensor(), which converts the image NDArray to Tensor NDArray.

## How to create a custom dataset

If the provided Datasets don't meet your requirements, you can also easily extend our dataset to create your own customized dataset.

Let's take CSVDataset, which can load a csv file, for example.

### Step 1: Prerequisites
For this example, we'll use [malicious_url_data.csv](https://github.com/incertum/cyber-matrix-ai/blob/master/Malicious-URL-Detection-Deep-Learning/data/url_data_mega_deep_learning.csv).

The CSV file has the following format.

| URL      | isMalicious |
| ----------- | ----------- |
| sample.url.good.com | 0 |
| sample.url.bad.com | 1  |

We'll also use the 3rd party [Apache Commons](https://commons.apache.org/) library to read the CSV file. To use the library, include the following dependency:

```
api group: 'org.apache.commons', name: 'commons-csv', version: '1.7'
```

### Step 2: Implementation
In order to extend the dataset, the following dependencies are required:

```
api "ai.djl:api:0.12.0"
api "ai.djl:basicdataset:0.12.0"
```

There are four parts we need to implement for CSVDataset.

1. Constructor and Builder

First, we need a private field that holds the CSVRecord list from the csv file.
We create a constructor and pass the CSVRecord list from builder to the class field.
For builder, we have all we need in `BaseBuilder` so we only need to include the two minimal methods as shown.
In the *build()* method, we take advantage of CSVParser to get the record of each CSV file and put them in CSVRecord list.

```java
public class CSVDataset extends RandomAccessDataset {

    private final List<CSVRecord> csvRecords;

    private CSVDataset(Builder builder) {
        super(builder);
        csvRecords = builder.csvRecords;
    }
    ...
    public static final class Builder extends BaseBuilder<Builder> {
        List<CSVRecord> csvRecords;

        @Override
        protected Builder self() {
            return this;
        }

        CSVDataset build() throws IOException {
            String csvFilePath = "path/malicious_url_data.csv";
            try (Reader reader = Files.newBufferedReader(Paths.get(csvFilePath));
                 CSVParser csvParser =
                    new CSVParser(
                        reader,
                        CSVFormat.DEFAULT
                            .withHeader("url", "isMalicious")
                            .withFirstRecordAsHeader()
                            .withIgnoreHeaderCase()
                            .withTrim())) {
                 csvRecords = csvParser.getRecords();
            }
            return new CSVDataset(this);
        }
    }

}
```

2. Prepare

As mentioned, in this example we are taking advantage of CSVParser to prepare the data for us. To prepare
the data on our own, we use the `prepare()` method. Normally here we would load or create any data 
for our dataset and then save it in one of the private fields previously created. This `prepare()` method
is called everytime we call `getData()` so in every case we want to only load the data once, we use a 
boolean variable called `prepared` to check if it has previously been loaded or prepared.

Since we don't have to prepare any data on our own for this example, we only have to override it.

```java
@Override
public void prepare(Progress progress) {}
```

There are great [examples](https://github.com/deepjavalibrary/djl/blob/master/basicdataset/src/main/java/ai/djl/basicdataset/nlp/AmazonReview.java)
in our [basicdataset](https://github.com/deepjavalibrary/djl/blob/master/basicdataset/src/main/java/ai/djl/basicdataset)
folder that show use cases for `prepare()`.



3. Getter

The getter returns a Record object which contains encoded inputs and labels.
Here, we use simple encoding to transform the url String to an int array and create a NDArray on top of it.
The reason why we use NDList here is that you might have multiple inputs and labels in different tasks.

```java
@Override
public Record get(NDManager manager, long index) {
    // get a CSVRecord given an index
    CSVRecord record = csvRecords.get(Math.toIntExact(index));
    NDArray datum = manager.create(encode(record.get("url")));
    NDArray label = manager.create(Float.parseFloat(record.get("isMalicious")));
    return new Record(new NDList(datum), new NDList(label));
}
```

4. Size

The number of records available to be read in this Dataset.
Here, we can directly use the size of the List<CSVRecord>.

```java
@Override
public long availableSize() {
    return csvRecords.size();
}
```

Done!
Now, you can use the CSVDataset with the following code snippet:

```java
CSVDataset dataset = new CSVDataset.Builder().setSampling(batchSize, false).build();
for (Batch batch : dataset.getData(model.getNDManager())) {
    // use head to get first NDArray
    batch.getData().head();
    batch.getLabels().head();
    ...
    // don't forget to close the batch in the end
    batch.close();
}
```

Full example code could be found in [CSVDataset.java](https://github.com/deepjavalibrary/djl/blob/master/docs/development/CSVDataset.java).
