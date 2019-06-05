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
package com.amazon.ai.metric;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

public class Metrics {

    private static final MetricValueComparator VALUE_COMPARATOR = new MetricValueComparator();

    private Map<String, List<Metric>> metrics;

    public Metrics() {
        metrics = new ConcurrentHashMap<>();
    }

    public void addMetric(Metric metric) {
        List<Metric> list = metrics.computeIfAbsent(metric.getMetricName(), v -> new ArrayList<>());
        list.add(metric);
    }

    public void addMetric(String name, Number value) {
        addMetric(new Metric(name, value));
    }

    public void addMetric(String name, Number value, String unit) {
        addMetric(new Metric(name, value, unit));
    }

    public List<Metric> getMetric(String name) {
        List<Metric> list = metrics.get(name);
        if (list == null) {
            return Collections.emptyList();
        }
        return list;
    }

    public Metric percentile(String metricName, int percent) {
        List<Metric> metric = metrics.get(metricName);
        if (metric == null || metrics.isEmpty()) {
            throw new IllegalArgumentException("Metric name not found: " + metricName);
        }

        List<Metric> list = new ArrayList<>(metric);
        list.sort(VALUE_COMPARATOR);
        int index = metric.size() * percent / 100;
        return list.get(index);
    }

    public double mean(String metricName, int percent) {
        List<Metric> metric = metrics.get(metricName);
        if (metric == null || metrics.isEmpty()) {
            throw new IllegalArgumentException("Metric name not found: " + metricName);
        }

        return metric.stream().collect(Collectors.averagingLong(m -> m.getValue().longValue()));
    }

    private static final class MetricValueComparator implements Comparator<Metric>, Serializable {

        private static final long serialVersionUID = 1L;

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
