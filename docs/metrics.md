# Metrics in DJL   
DJL comes with utility classes to make it easy to capture performance and other metrics during runtime that could be used to analyze and monitor inference and/or training performance and stability. [Metrics](../api/src/main/java/ai/djl/metric/Metrics.java) is the class that enables that. It is built as a collection of individual [Metric](../api/src/main/java/ai/djl/metric/Metric.java) classes.

As a container for individual metrics class **Metrics** stores them as a time series data so that metric-vs-timeline analysis could be performed. It also provides convenient statistical methods for getting aggregated information, such as _mean_ and _percentile_.

DJL uses Metrics collection to store key performance indicators (KPIs) during inference and training runs such as various latencies, CPU and GPU memory consumption, losses, etc. They can be accessed out of the box if you utilize DJL built in classes in your applications. [See more](#metrics-out-of-the-box) below for examples of these built-in classes in DJL classes. On top of that DJL provides an easy way to keep track of your own metrics in a similar manner. [See more](#user-defined-metrics) on that further below.

## Metrics out of the box
A lot of DJL classes keep track of relevant quantitative and qualitative metrics and provide easy access to them. For example, if there is an application that uses Predictor to serve inference requests here is how to access mean and p90 latency of individual requests:
```java
// load image in which objects need to be detected
URL url = new URL("https://s3.amazonaws.com/images.pdpics.com/preview/3033-bicycle-rider.jpg");
BufferedImage img = ImageIO.read(url);

// load model for SingleShotDetector
Map<String, String> criteria = new HashMap<>();
criteria.put("size", "512");
criteria.put("backbone", "resnet50");
criteria.put("dataset", "voc");
ZooModel<BufferedImage, DetectedObjects> model = MxModelZoo.SSD.loadModel(criteria);
Predictor<BufferedImage, DetectedObjects> predictor = model.newPredictor();

// instantiate metrics so that Predictor would record its performance
Metrics metrics = new Metrics();
predictor.setMetrics(metrics);

// run object detection 100 times to generate time series in metrics collection
for (int i = 0; i < 100; ++i) {
    predictor.predict(img);
}

// inspect mean and p90 latencies of 100 inference requests
double inferenceMean = metrics.mean("Inference");
Number inferenceP90 = metrics.percentile("Inference", 90).getValue();
```

Please note an important detail: in order to ensure that DJL objects will capture _**metrics**_ - they (metrics) have to be instantiated manually before engaging functionality of DJL objects. Underlying DL engine usually optimizes execution flow of the model's forward and backward passes. As a result of that multiple parts of model's graph could be run in parallel for better performance. That is an upside of such optimization. The downside is that it becomes tricky to measure metrics like latency because actual pass through the model happens at a time that is different from when DJL calls engine's _forward_ method. In order to compensate for such an obstacle DL engine provides a mechanism to ensure that a call to forward pass, for example, will not return until that pass has been actually executed by the engine at a time of a call, without delays. As this means less optimal execution flow for model's graph - this is exposed as optional functionality in DJL. By default, when no metrics object is provided for DJL class - no metrics will be recorded, in order to not "mess" with execution flow optimizations. If metrics are needed - they need to be instantiated from outside of DJL object and passed to it. DJL object will use this Metrics to record its relevant KPIs. After DJL object's function returns all recorded metrics are recorded and exposed.

## User defined metrics
DJL approach to out of the box metrics has another upside to it: if application/service needs to records its own metrics and KPIs - it can use exactly the same approach and similar constructs. Here is a very simplistic example of how application can measure and record its own custom metric of latency of a particular piece of code.  
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

## More references
For more examples of how metrics could be used as well as convenient utils provided by DJL please refer to:
- [MemoryUtils](../examples/src/main/java/ai/djl/examples/util/MemoryUtils.java) for how memory consumption metrics could be captured
- [MxTrainer](../mxnet/engine/src/main/java/ai/djl/mxnet/engine/MxTrainer.java) for some of the metrics that are captured during training
- [BasePredictor](../api/src/main/java/ai/djl/inference/BasePredictor.java) for some of the metrics that are captured during inference time