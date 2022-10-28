# DJL timeseries package applied on M5forecasting dataset

The timeseries package contains components and tools for building time series models and
inferring on pretrained models using DJL.

Now it contains:

- Translator for preprocess and postprocess data, and also includes the corresponding data transform modules.
- Components for building and training new probabilistic prediction models.
- Time series data loading and processing.
- A M5-forecast dataset.
- Two pre-trained model.
- A pre-built trainable model (DeepAR).

## M5 Forecasting data

[M5 Forecasting competition]([M5 Forecasting - Accuracy | Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/overview/description)) 
goal is to forecast future sales at Walmart based on hierarchical sales in the states of California,
Texas, and Wisconsin. It provides information on daily sales, product attributes, prices, and calendars.

> Notes: Taking into account the model training performance, we sum the sales every 7 days,  
> coarse-grained the data, so that the model can better learn the time series information.
> **After downloading the dataset from [M5 Forecasting competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/overview/description),
> you can use our [script](https://gist.github.com/Carkham/a5162c9298bc51fec648a458a3437008)
> to get coarse-grained data. This script will create "weekly_xxx.csv" files representing weekly
> data in the dataset directory you specify.**

## DeepAR model

DeepAR forecasting algorithm is a supervised learning algorithm for forecasting scalar
(one-dimensional) time series using recurrent neural networks (RNN).

Unlike traditional time series forecasting models, DeepAR estimates the future probability
distribution of time series based on the past. In retail businesses, probabilistic demand
forecasting is critical to delivering the right inventory at the right time and in the right place.

Therefore, we choose the sales data set in the real scene as an example to describe how to use
the timeseries package for forecasting

### Metrics

We use the following metrics to evaluate the performance of the DeepAR model in the
[M5 Forecasting competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/overview/description).

```
> [INFO ] - metric: Coverage[0.99]:	0.92
> [INFO ] - metric: Coverage[0.67]:	0.51
> [INFO ] - metric: abs_target_sum:	1224665.00
> [INFO ] - metric: abs_target_mean:	10.04
> [INFO ] - metric: NRMSE:	0.84
> [INFO ] - metric: RMSE:	8.40
> [INFO ] - metric: RMSSE:	1.00
> [INFO ] - metric: abs_error:	14.47
> [INFO ] - metric: QuantileLoss[0.67]:	18.23
> [INFO ] - metric: QuantileLoss[0.99]:	103.07
> [INFO ] - metric: QuantileLoss[0.50]:	9.49
> [INFO ] - metric: QuantileLoss[0.95]:	66.69
> [INFO ] - metric: Coverage[0.95]:	0.87
> [INFO ] - metric: Coverage[0.50]:	0.33
> [INFO ] - metric: MSE:	70.64
```

As you can see, our pretrained model has some effect on the data prediction of item value. And
some metrics can basically meet expectations. For example, **RMSSE**, which is a measure of the
relative error between the predicted value and the actual value. 1.00 means that the model can
reflect the changes of the time series data to a certain extent.

## Run the M5 Forecasting example

### Define your dataset

In order to realize the preprocessing of time series data, we define the `TimeSeriesData` as
the input of the Translator, which is used to store the feature fields and perform corresponding
transformations.

So for your own dataset, you need to customize the way you get the data and put it into
`TimeSeriesData` as the input to the translator.

For M5 dataset we have:

```java
private static final class M5Dataset implements Iterable<NDList>, Iterator<NDList> {
		
    	// coarse-grained data
        private static String fileName = "weekly_sales_train_evaluation.csv";

        private NDManager manager;
        private List<Feature> target;
        private List<CSVRecord> csvRecords;
        private long size;
        private long current;

        M5Dataset(Builder builder) {
            manager = builder.manager;
            target = builder.target;
            try {
                prepare(builder);
            } catch (Exception e) {
                throw new AssertionError(
                        String.format("Failed to read m5-forecast-accuracy/%s.", fileName), e);
            }
            size = csvRecords.size();
        }

    	/** Load data into CSVRecords */
        private void prepare(Builder builder) throws IOException {
            URL csvUrl = builder.root.resolve(fileName).toUri().toURL();
            try (Reader reader =
                    new InputStreamReader(
                            new BufferedInputStream(csvUrl.openStream()), StandardCharsets.UTF_8)) {
                CSVParser csvParser = new CSVParser(reader, builder.csvFormat);
                csvRecords = csvParser.getRecords();
            }
        }

        @Override
        public boolean hasNext() {
            return current < size;
        }

        @Override
        public NDList next() {
            NDList data = getRowFeatures(manager, current, target);
            current++;
            return data;
        }

        public static Builder builder() {
            return new Builder();
        }

    	/** Get string data of selected cell from index row in CSV file and create NDArray to save  */
        private NDList getRowFeatures(NDManager manager, long index, List<Feature> selected) {
            DynamicBuffer bb = new DynamicBuffer();
            for (Feature feature : selected) {
                String name = feature.getName();
                String value = getCell(index, name);
                feature.getFeaturizer().featurize(bb, value);
            }
            FloatBuffer buf = bb.getBuffer();
            return new NDList(manager.create(buf, new Shape(bb.getLength())));
        }

        private String getCell(long rowIndex, String featureName) {
            CSVRecord record = csvRecords.get(Math.toIntExact(rowIndex));
            return record.get(featureName);
        }

        @Override
        public Iterator<NDList> iterator() {
            return this;
        }

        public static final class Builder {

            NDManager manager;
            List<Feature> target;
            CSVFormat csvFormat;
            Path root;

            Builder() {
                csvFormat =
                        CSVFormat.DEFAULT
                                .builder()
                                .setHeader()
                                .setSkipHeaderRecord(true)
                                .setIgnoreHeaderCase(true)
                                .setTrim(true)
                                .build();
                target = new ArrayList<>();
                for (int i = 1; i <= 277; i++) {
                    target.add(new Feature("w_" + i, true));
                }
            }

            public Builder setRoot(Path root) {
                this.root = root;
                return this;
            }

            public Builder setManager(NDManager manager) {
                this.manager = manager;
                return this;
            }

            public M5Dataset build() {
                return new M5Dataset(this);
            }
        }
    }
```

### Prepare dataset

Set your own dataset path.

```java
Path m5ForecastFile = Paths.get("/YOUR PATH/m5-forecasting-accuracy");
NDManager manager = NDManager.newBaseManager();
M5Dataset dataset = M5Dataset.builder().setManager(manager).setRoot(m5ForecastFile).build();
```

### Config your translator

`DeepARTranslator` provides support for data preprocessing and postprocessing for probabilistic
prediction models. Referring to GluonTS, our translator can perform corresponding preprocessing
on `TimeseriesData` containing data according to different parameters to obtain the input of
the network model. And post-processing the output of the network to get the prediction result.

For DeepAR models, you must set the following arguments.

```java
Logger logger = LoggerFactory.getLogger(TimeSeriesDemo.class);
String freq = "W";
int predictionLength = 4;
LocalDateTime startTime = LocalDateTime.parse("2011-01-29T00:00");

Map<String, Object> arguments = new ConcurrentHashMap<>();

arguments.put("prediction_length", predictionLength);
arguments.put("freq", freq); // The predicted frequency contains units and values

// Parameters from DeepAR in GluonTS
arguments.put("use_" + FieldName.FEAT_DYNAMIC_REAL.name().toLowerCase(), false); 
arguments.put("use_" + FieldName.FEAT_STATIC_CAT.name().toLowerCase(), false);
arguments.put("use_" + FieldName.FEAT_STATIC_REAL.name().toLowerCase(), false);
```

For any other GluonTS model, you can quickly develop your own translator using the classes
in `transform` modules (etc. `TransformerTranslator`).

### Load your own model from the local file system

At this step, you need to construct the `Criteria` API, which is used as search criteria to look
for a ZooModel. In this application, you can customize your local pretrained model path
(local directory or an archive file containing .`params` and `symbol.json`.)
with .`optModelPath()`. The following code snippet loads the model with the file
path: `/YOUR PATH/deepar.zip` .

```java
DeepARTranslator translator = DeepARTranslator.builder(arguments).build();
Criteria<TimeSeriesData, Forecast> criteria =
        Criteria.builder()
                .setTypes(TimeSeriesData.class, Forecast.class)
                .optModelPath(Paths.get("/YOUR PATH/deepar.zip"}))
                .optTranslator(translator)
                .optProgress(new ProgressBar())
                .build();
```

### Inference

Now, you are ready to used the model bundled with the translator created above to run inference.

Since we need to generate features based on dates and make predictions with reference to the
context, for each `TimeSeriesData` you must set the values of its **`StartTime`** and **`TARGET`** fields.

```java
try (ZooModel<TimeSeriesData, Forecast> model = criteria.loadModel();
             Predictor<TimeSeriesData, Forecast> predictor = model.newPredictor()) {
    data = dataset.next();
    NDArray array = data.singletonOrThrow();
    TimeSeriesData input = new TimeSeriesData(10);
    input.setStartTime(startTime); // start time of prediction
    input.setField(FieldName.TARGET, array); // target value through whole context length
    Forecast forecast = predictor.predict(input);
    saveResult(forecast); // save result and plot it with python.
    }
}
```

### Results

The `Forecast` are objects that contain all the sample paths in the form of `NDArray`
with dimension `(numSamples, predictionLength)`, the start date of the forecast. You can access
all these information by simply invoking the corresponding function.

You can summarize the sample paths by computing, including the mean and quantile, for each step
in the prediction window.

```java
logger.info("Mean of the prediction windows:\n" + forecast.mean().toDebugString());
logger.info("0.5-quantile(Median) of the prediction windows:\n" + forecast.quantile("0.5").toDebugString());
```

```
> [INFO ] - Mean of the prediction windows:
> ND: (4) cpu() float32
> [5.97, 6.1 , 5.9 , 6.11]
>
> [INFO ] - 0.5-quantile(Median) of the prediction windows:
> ND: (4) cpu() float32
> [6., 5., 5., 6.]
```

We visualize the forecast result with mean, prediction intervals, etc.

![m5_forecast_0](https://resources.djl.ai/images/timeseries/m5_forecast_0.jpg)

### Metrics

Here we compute aggregate performance metrics in the
[source code](../../examples/src/main/java/ai/djl/examples/inference/timeseries/M5ForecastingDeepAR.java)
