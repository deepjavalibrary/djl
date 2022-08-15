# TimeSeries support

This module contains the time series model support extension with [gluon-ts](https://github.com/awslabs/gluon-ts).

Right now, the package provides the `BaseTimeSeriesTranslator` and transform package that allows you to do inference from your pre-trained time series model.

The following pseudocode demonstrates how to create a `DeepARTranslator` with `arguments`.

```java
	Map<String, Object> arguments = new ConcurrentHashMap<>();
	arguments.put("prediction_length", 28);
	arguments.put("use_feat_dynamic_real", false);
	DeepARTranslator.Builder builder = DeepARTranslator.builder(arguments);
	DeepARTranslator translator = builder.build();
```

If you want to customize your own time series model translator, you can easily use the transform package for your data preprocess.

See [examples](./src/main/java/ai/djl/timeseries/examples) for more details.

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