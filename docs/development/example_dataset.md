## Example CSV Dataset

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
api "ai.djl:api:0.21.0"
api "ai.djl:basicdataset:0.21.0"
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
                            .builder()
                            .setHeader("url", "isMalicious")
                            .setSkipHeaderRecord(true)
                            .setIgnoreHeaderCase(true)
                            .setTrim(true)
                            .build())) {
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
