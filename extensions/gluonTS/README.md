# GluonTS support

This module contains the GluonTS support extension with [gluon-ts](https://github.com/awslabs/gluon-ts).

Right now, the package provides the `BaseGluonTSTranslator` and transform package that allows you do inference from your pre-trained model with GluonTS.

The following pseudocode demonstrates how to create a `DeepARTranslator` with `arguments`.

```java
	Map<String, Object> arguments = new ConcurrentHashMap<>();
	arguments.put("prediction_length", 28);
	arguments.put("use_feat_dynamic_real", false);
	DeepARTranslator.Builder builder = DeepARTranslator.builder(arguments);
	DeepARTranslator translator = builder.build();
```

If you want to customize your own GluonTS model translator, you can easily use the transform package for your data preprocess.

See [examples](./src/main/java/ai/djl/gluonTS/examples) for more details.

We plan to add the following features in the future:

- a `GluonTSDataset`class to support creating data entry and transforming raw csv data like in GluonTS.
- Many GluonTS models that can be trained in djl.
- ......

## Documentation

You can build the latest javadocs locally using the following command:

```sh
./gradlew javadoc
```

The javadocs output is built in the `build/doc/javadoc` folder.

Some of our comments refer to gluonts, please check the [GluonTS documentation](https://ts.gluon.ai/stable/) for more information.