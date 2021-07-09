# Metrics

Deep Java Library (DJL) comes with utility classes to make it easy to capture performance metrics
and other metrics during runtime. These metrics can be used to analyze and monitor inference,
training performance, and stability. [Metrics](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/metric/Metrics.html)
is the class that enables collecting metrics information. It is built as a collection of individual
[Metric](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/metric/Metric.html) classes.

As a container for individual metrics classes, **Metrics** stores them as time series data so that
metric-vs-timeline analysis could be performed. It also provides convenient statistical methods for
getting aggregated information, such as _mean_ and _percentile_.

DJL uses the Metrics collection to store key performance indicators (KPIs) during inference and
training runs. These KPIs include various latencies, CPU and GPU memory consumption, losses, etc.
They can be accessed if you utilize built-in DJL classes in your applications. For examples of
these built-in classes, see [Use metrics with DJL](#use-metrics-with-djl). 
DJL also provides an easy way to keep track of your own metrics. For more information,
see [User defined metrics](#user-defined-metrics).

## Use metrics with DJL
Many DJL classes keep track of relevant quantitative and qualitative metrics. DJL provides easy access to these metrics.

For example, if there is an application that uses Predictor to serve inference requests, the
following code can be used to access mean and p90 latency of individual requests:

```java
// load the image in which objects need to be detected
URL url = new URL("https://s3.amazonaws.com/images.pdpics.com/preview/3033-bicycle-rider.jpg");
Image img = ImageFactory.getInstance().fromURL(url);

// load the model for SingleShotDetector
Criteria<Image, DetectedObjects> criteria = Criteria.builder()
    .setTypes(Image.class, DetectedObjects.class)
    .optArtifactId("ssd")
    .optFilter("backbone", "resnet50")
    .optFilter("dataset", "voc")
    .optFilter("size", "512")
    .build();

ZooModel<Image, DetectedObjects> model = criteria.loadModel();
Predictor<Image, DetectedObjects> predictor = model.newPredictor();

// instantiate metrics so that Predictor can record its performance
Metrics metrics = new Metrics();
predictor.setMetrics(metrics);

// run object detection 100 times to generate a time series in metrics collection
for (int i = 0; i < 100; ++i) {
    predictor.predict(img);
}

// inspect mean and p90 latencies of 100 inference requests
double inferenceMean = metrics.mean("Inference");
Number inferenceP90 = metrics.percentile("Inference", 90).getValue();
```

In order to ensure that DJL objects will capture _**metrics**_, the metrics have to be instantiated manually before engaging the functionality of DJL objects. The underlying deep learning engine optimizes the execution flow of the model's forward and backward passes. Because of this, multiple parts of the model's graph can be run in parallel for better performance. 

The downside of this optimization is that it becomes tricky to measure metrics like latency.  Measurement is impacted because the actual pass through the model happens at a different time than when DJL calls the engine's _forward_ method. 

In order to compensate for this, the deep learning engine provides a mechanism to ensure that a call to forward pass, for example, will not return until that pass has been executed by the engine. Because it creates a less optimal execution flow for the model's graph, this is optional functionality in DJL. By default, when no metrics object is provided for DJL class, no metrics will be recorded. This avoids an impact on execution flow optimizations. If metrics are needed, they must be instantiated from outside of the DJL object and passed in to it. The DJL object will use this Metrics object to record its relevant KPIs. After the DJL object's function returns, all recorded metrics are recorded and exposed.

## User defined metrics
The DJL approach to out of the box metrics has another benefit. If an application or service needs to record its own metrics and KPIs, it can use the same approach with similar constructs. 

The following is a simplistic example showing how an application can measure and record its own custom latency metric for a particular piece of code:

```java
Metrics metrics = new Metrics();
long begin = System.nanoTime();

int i = 0;
Random r = new Random();
while (i++ < 100) {
    int sleepInterval = r.nextInt(100);
    Thread.sleep(sleepInterval);
    metrics.addMetric("single_latency", sleepInterval, "ms");
}

long end = System.nanoTime();
metrics.addMetric("end_to_end_latency", (end-begin) / 1_000_000f, "ms");
```

## More information

For more examples of metrics use, as well as convenient utilities provided by DJL, see:

- [MemoryTrainingListener](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/training/listener/MemoryTrainingListener.html) for memory consumption metrics
- [Trainer](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/training/Trainer.html) for metrics during training
- [Predictor](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/inference/Predictor.html) for metrics during inference

