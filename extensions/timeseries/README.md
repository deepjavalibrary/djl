# TimeSeries support

This module contains the time series model support extension with [gluon-ts](https://github.com/awslabs/gluon-ts).

Right now, the package provides the `BaseTimeSeriesTranslator` and transform package that allows you to do inference from your pre-trained time series model.

## Module structure

### Forecast

An abstract class representing the forecast result.

It contains the distribution of the results, the start date of the forecast, the frequency of the time series, etc. User can get all these information by simply invoking the corresponding attribute.

- `SampleForecast` extends the `Forecast` that contain all the sample paths in the form of `NDArray`. User can query the prediction results by accessing the data in the samples.

### TimeSeriesData

The data entry for managing timing data in preprocessing as an input to the transform method. It contains a key-value pair list mapping from the time series name field to `NDArray`.

### dataset

- FieldName -- The name field for time series data including START, TARGET, and so on.

### timefeature

This module contains all the methods for generating time features from the predicted frequency.

- Lag -- Generates a list of lags that are appropriate for the frequency.
- TimeFeature -- Generates a list of time features that are appropriate for the frequency.

### transform

In general, it gets the `TimeSeriesData` and transform it to another `TimeSeriesData` that can possibly contain more fields. It can be done by defining a set of of "actions" to the raw dataset in training or just invoking at translator in inference.

This action usually create some additional features or transform an existing feature.

#### convert

- Convert -- Convert the array shape to the preprocessing. 
- VstackFeatures.java -- vstack the inputs name field of the `TimeSeriesData`. We make it implement the `TimeSeriesTransform` interface for **training feature.**

#### feature

- Feature -- Add time features to the preprocessing. 
- AddAgeFeature -- Creates the `FEAT_DYNAMIC_AGE` name field in the `TimeSeriesData`. Adds a feature that its value is small for distant past timestamps and it monotonically increases the more we approach the current timestamp. We make it implement the `TimeSeriesTransform` interface for **training feature.**
- AddObservedValueIndicator -- Creates the `OBSERVED_VALUES` name field in the `TimeSeriesData`. Adds a feature that equals to 1 if the value is observed and 0 if the value is missing. We make it implement the `TimeSeriesTransform` interface for **training feature.**
- AddTimeFeature -- Creates the `FEAT_TIME` name field in the `TimeSeriesData`. Adds a feature that its value is based on the different prediction frequencies. We make it implement the `TimeSeriesTransform` interface for **training feature.**

#### field

- Field -- Process key-value data entry to the preprocessing. It usually add or remove the feature in the `TimeSeriesData`.
- RemoveFields -- Remove the input name field. We make it implement the `TimeSeriesTransform` interface for **training feature.**
- SelectField -- Only keep input name fields. We make it implement the `TimeSeriesTransform` interface for **training feature.**
- SetField -- Set the input name field with `NDArray`. We make it implement the `TimeSeriesTransform` interface for **training feature.**

#### split

- Split -- Split time series data for training and inferring to the preprocessing.
- InstanceSplit -- Split time series data with the slice from `Sampler` for training and inferring to the preprocessing. We make it implement the `TimeSeriesTransform` interface for **training feature.**

### InstanceSampler

Sample index for splitting based on training or inferring.

`PredictionSampler` extends `InstanceSampler` for the prediction including test and valid. It would return the end of the time series bound as the dividing line between the future and past.

### translator

Existing time series model translators and corresponding factories. Now we have developed `DeepARTranslator` and `TransformerTranslator` for users.

The following pseudocode demonstrates how to create a `DeepARTranslator` with `arguments`.

```java
	Map<String, Object> arguments = new ConcurrentHashMap<>();
	arguments.put("prediction_length", 28);
	arguments.put("use_feat_dynamic_real", false);
	DeepARTranslator.Builder builder = DeepARTranslator.builder(arguments);
	DeepARTranslator translator = builder.build();
```

If you want to customize your own time series model translator, you can easily use the transform package for your data preprocess.

See [examples](../src/main/java/ai/djl/timeseries/examples) for more details.

We plan to add the following features in the future:

- a `TimeSeriesDataset`class to support creating data entry and transforming raw csv data like in TimeSeries.
- Many time series models that can be trained in djl.
- ......

## Documentation

You can build the latest javadocs locally using the following command:

```sh
./gradlew javadoc
```

The javadocs output is built in the `build/doc/javadoc` folder.