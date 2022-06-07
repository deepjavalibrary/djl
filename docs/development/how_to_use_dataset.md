# Dataset Creation

The [Dataset](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/training/dataset/Dataset.html) in DJL represents both the raw data and the data loading process.
For this reason, training in DJL usually requires that your data be implemented through using a dataset class.
You can choose to use one of the well-known datasets we have [built in](../dataset.md).
Or, you can create a custom dataset.

## Dataset Helpers

There are a number of helpers provided by DJL to make it easy to create custom datasets.
If a helper is available, it can make it easier to implement the dataset then building it from scratch:

### CV

- [ImageDataset](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/cv/ImageDataset.html) - A abstract dataset to create a dataset where the input is an image such as image classification, object detection, and image segmentation
- [ImageClassificationDataset](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/cv/classification/ImageClassificationDataset.html) - An abstract dataset for image classification
- [AbstractImageFolder](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/cv/classification/AbstractImageFolder.html) - An abstract dataset for loading images in a folder structure. Usually you want the ImageFolderDataset.
- [ImageFolder](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/cv/classification/ImageFolder.html) - A dataset for loading image folders stored in a folder structure
- [ObjectDetectionDataset](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/cv/ObjectDetectionDataset.html) - An abstract dataset for object detection

### NLP

- [TextDataset](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/nlp/TextDataset.html) - An abstract dataset for NLP where either the input or labels are text-based.
- [TextData](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/utils/TextData.html) - A utility for managing the text within a dataset

### Tabular

- [CsvDataset](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/tabular/CsvDataset.html) - An dataset for loading data from a .csv file
- [TabularDataset](https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/tabular/TabularDataset.html) - An abstract dataset for loading tabular data with rows and feature columns
- [TablesawDataset](https://javadoc.io/doc/ai.djl.tablesaw/tablesaw/latest/ai/djl/tablesaw/TablesawDataset.html) - An dataset for loading from [Tablesaw](https://jtablesaw.github.io/tablesaw/)

## Custom Datasets

If none of the provided datasets meet your requirements, you can also easily customize you own dataset in a custom class.
While technically the dataset must only implement [`Dataset`](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/training/dataset/Dataset.html), it is best to instead extend [`RandomAccessDataset`](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/training/dataset/RandomAccessDataset.html).
It manages data randomization and provides comprehensive data loading functionality.

The `RandomAccessDataset` is based on making your data records into a list where each record has an index.
Then, it only needs to know how many records there are and how to load each record giving its index.

As part of implementing the dataset, there are two methods that must be defined:

- `Record get(NDManager manager, long index)` - Returns the record (both input data and output label) for a particular index
- `long availableSize()` - Returns the number of records in the dataset

In addition, the dataset should also have a nested builder class to contain details on how to load the dataset.
The builder would extend `RandomAccessDataset.BaseBuilder`.
This provides an avenue to modify how RandomAccessDataset loads the data.
You can also add your own options into the builder.
For an example of how this would look like, see [`ImageFolder.Builder`](https://github.com/deepjavalibrary/djl/blob/master/basicdataset/src/main/java/ai/djl/basicdataset/cv/classification/ImageFolder.java).

You can also view this example of creating a [new CSV dataset](example_dataset.md).

Many of the abstract dataset helpers above also extend `RandomAccessDataset`.
When using them, most of the same information applies.
You may be asked to implement slightly different methods depending on the particular extended class.
You will also want to extend that classes `BaseBuilder` instead of the one found in `RandomAccessDataset` to get the additional data loading options from the helper.

If you create a new dataset for public dataset, consider contributing that dataset back to DJL for others to use.
You can follow [these instructions](add_dataset_to_djl.md) for adding it.