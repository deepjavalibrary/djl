/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.metric;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * A collection of {@link Metric} objects organized by metric name.
 *
 * <p>{@code Metric} is a utility class that is used in the {@link ai.djl.training.Trainer} and
 * {@link ai.djl.inference.Predictor} to capture performance and other metrics during runtime.
 *
 * <p>It is built as a collection of individual {@link Metric} classes. As a container for
 * individual metrics classes, {@code Metrics} stores them as time series data so that
 * metric-vs-timeline analysis can be performed. It also provides convenient statistical methods for
 * getting aggregated information, such as mean and percentile. The metrics is used to store key
 * performance indicators (KPIs) during inference and training runs. These KPIs include various
 * latencies, CPU and GPU memory consumption, losses, etc.
 *
 * <p>For more details about using the metrics, see the <a
 * href="https://github.com/deepjavalibrary/djl/blob/master/docs/how_to_collect_metrics.md">metrics
 * tutorial</a>.
 */
public class Metrics {

    private static final MetricValueComparator VALUE_COMPARATOR = new MetricValueComparator();

    private Map<String, List<Metric>> metrics;

    /** Constructs an empty {@code Metrics} instance. */
    public Metrics() {
        metrics = new ConcurrentHashMap<>();
    }

    /**
     * Adds a {@link Metric} to the collection.
     *
     * @param metric the {@link Metric} to be added
     */
    public void addMetric(Metric metric) {
        List<Metric> list =
                metrics.computeIfAbsent(
                        metric.getMetricName(),
                        v -> Collections.synchronizedList(new ArrayList<>()));
        list.add(metric);
    }

    /**
     * Adds a {@code Metric} given the metric's {@code name} and {@code value}.
     *
     * @param name the metric name
     * @param value the metric value
     */
    public void addMetric(String name, Number value) {
        addMetric(new Metric(name, value));
    }

    /**
     * Adds a {@code Metric} given the metric's {@code name}, {@code value}, and {@code unit}.
     *
     * @param name the metric name
     * @param value the metric value
     * @param unit the metric unit
     */
    public void addMetric(String name, Number value, String unit) {
        addMetric(new Metric(name, value, unit));
    }

    /**
     * Returns {@code true} if the metrics object has a metric with the given name.
     *
     * @param name the name to check for
     * @return {@code true} if the metrics object has a metric with the given name
     */
    public boolean hasMetric(String name) {
        return metrics.containsKey(name);
    }

    /**
     * Returns all {@link Metric}s with the specified metric name.
     *
     * @param name the name of the metric
     * @return a list of {@link Metric} with the specified metric name
     */
    public List<Metric> getMetric(String name) {
        List<Metric> list = metrics.get(name);
        if (list == null) {
            return Collections.emptyList();
        }
        return list;
    }

    /**
     * Returns a set of {@link String} metric names.
     *
     * @return a set of {@link String} metric names
     */
    public Set<String> getMetricNames() {
        return metrics.keySet();
    }

    /**
     * Returns the latest {@link Metric} with the specified metric name.
     *
     * @param name the name of the metric
     * @return the {@link Metric} with the specified metric name
     * @throws IllegalArgumentException if the given name is not found
     */
    public Metric latestMetric(String name) {
        List<Metric> list = metrics.get(name);
        if (list == null || list.isEmpty()) {
            throw new IllegalArgumentException("Could not find metric: " + name);
        }
        return list.get(list.size() - 1);
    }

    /**
     * Returns a percentile {@link Metric} object for the specified metric name.
     *
     * @param metricName the name of the metric
     * @param percentile the percentile
     * @return the {@link Metric} object at specified {@code percentile}
     */
    public Metric percentile(String metricName, int percentile) {
        List<Metric> metric = metrics.get(metricName);
        if (metric == null || metrics.isEmpty()) {
            throw new IllegalArgumentException("Metric name not found: " + metricName);
        }

        List<Metric> list = new ArrayList<>(metric);
        list.sort(VALUE_COMPARATOR);
        int index = metric.size() * percentile / 100;
        return list.get(index);
    }

    /**
     * Returns the average value of the specified metric.
     *
     * @param metricName the name of the metric
     * @return the average value of the specified metric
     */
    public double mean(String metricName) {
        List<Metric> metric = metrics.get(metricName);
        if (metric == null || metrics.isEmpty()) {
            throw new IllegalArgumentException("Metric name not found: " + metricName);
        }

        return metric.stream().collect(Collectors.averagingDouble(m -> m.getValue().doubleValue()));
    }

    /** Comparator based on {@code Metric}'s value field. */
    private static final class MetricValueComparator implements Comparator<Metric>, Serializable {

        private static final long serialVersionUID = 1L;

        /** {@inheritDoc} */
        @Override
        public int compare(Metric o1, Metric o2) {
            Number n1 = o1.getValue();
            Number n2 = o2.getValue();
            if (n1 instanceof Double || n1 instanceof Float) {
                return Double.compare(n1.doubleValue(), n2.doubleValue());
            }
            return Long.compare(n1.longValue(), n2.longValue());
        }
    }
}
