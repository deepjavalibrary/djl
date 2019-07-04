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
package software.amazon.ai.metric;

/** A class representing a {@code Metric} item. */
public class Metric {

    private String metricName;
    private Number value;
    private String unit;
    private long timestamp;

    /**
     * Constructs a {@code Metric} instance with specified {@code metricName} and <code>
     * value</code>.
     *
     * @param metricName metric name
     * @param value metric value
     */
    public Metric(String metricName, Number value) {
        this(metricName, value, "count");
    }

    /**
     * Constructs a {@code Metric} instance with specified {@code metricName}, <code>value
     * </code> and {@code unit}.
     *
     * @param metricName metric name
     * @param value metric value
     * @param unit metric unit
     */
    public Metric(String metricName, Number value, String unit) {
        this.metricName = metricName;
        this.value = value;
        this.unit = unit;
        timestamp = System.currentTimeMillis();
    }

    /**
     * Returns name of the {@code Metric}.
     *
     * @return metric name
     */
    public String getMetricName() {
        return metricName;
    }

    /**
     * Returns value of the {@code Metric}.
     *
     * @return metric value
     */
    public Number getValue() {
        return value;
    }

    /**
     * Returns unit of the {@code Metric}.
     *
     * @return metric unit
     */
    public String getUnit() {
        return unit;
    }

    /**
     * Returns timestamp of the {@code Metric}.
     *
     * @return metric timestamp
     */
    public long getTimestamp() {
        return timestamp;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return metricName + '.' + unit + ':' + value + "|#timestamp:" + timestamp;
    }
}
