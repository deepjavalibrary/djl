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

import com.google.gson.annotations.SerializedName;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A class representing a single recorded {@code Metric} value.
 *
 * @see Metrics
 */
public class Metric {

    private static final Pattern PATTERN =
            Pattern.compile(
                    "\\s*([\\w\\s]+)\\.([\\w\\s]+):([0-9\\-,.e]+)(?>\\|#([^|]*))?(?>\\|(\\d+))?");

    private static final Dimension HOST = new Dimension("Host", getLocalHostName());

    @SerializedName("MetricName")
    private String metricName;

    @SerializedName("Value")
    private String value;

    @SerializedName("Unit")
    private String unit;

    @SerializedName("Dimensions")
    private Dimension[] dimensions;

    @SerializedName("Timestamp")
    private String timestamp;

    /**
     * Constructs a {@code Metric} instance with the specified {@code metricName} and <code>
     * value</code>.
     *
     * @param metricName the metric name
     * @param value the metric value
     */
    public Metric(String metricName, Number value) {
        this(metricName, value, Unit.COUNT);
    }

    /**
     * Constructs a {@code Metric} instance with the specified {@code metricName}, <code>value
     * </code>, and {@code unit}.
     *
     * @param metricName the metric name
     * @param value the metric value
     * @param unit the metric unit
     */
    public Metric(String metricName, Number value, Unit unit) {
        this(metricName, value.toString(), unit.getValue(), null, HOST);
    }

    /**
     * Constructs a {@code Metric} instance with the specified {@code metricName}, <code>value
     * </code>, and {@code unit}.
     *
     * @param metricName the metric name
     * @param value the metric value
     * @param unit the metric unit
     * @param dimensions the metric dimensions
     */
    public Metric(String metricName, Number value, Unit unit, Dimension... dimensions) {
        this(metricName, value.toString(), unit.getValue(), null, dimensions);
    }

    /**
     * Constructs a new {@code Metric} instance.
     *
     * @param metricName the metric name
     * @param value the metric value
     * @param unit the metric unit
     * @param timestamp the metric timestamp
     * @param dimensions the metric dimensions
     */
    private Metric(
            String metricName,
            String value,
            String unit,
            String timestamp,
            Dimension... dimensions) {
        this.metricName = metricName;
        this.unit = unit;
        this.value = value;
        this.timestamp = timestamp;
        this.dimensions = dimensions;
    }

    /**
     * Returns a copy of the metric with a new name.
     *
     * @param name the new metric name
     * @return a copy of the metric
     */
    public Metric copyOf(String name) {
        return new Metric(name, value, unit, timestamp, dimensions);
    }

    /**
     * Returns the name of the {@code Metric}.
     *
     * @return the metric name
     */
    public String getMetricName() {
        return metricName;
    }

    /**
     * Returns the int value of the {@code Metric}.
     *
     * @return the metric value in int
     */
    public Double getValue() {
        return Double.valueOf(value);
    }

    /**
     * Returns the unit of the {@code Metric}.
     *
     * @return the metric unit
     */
    public Unit getUnit() {
        return Unit.fromValue(unit);
    }

    /**
     * Returns the timestamp of the {@code Metric}.
     *
     * @return the metric timestamp
     */
    public String getTimestamp() {
        return timestamp;
    }

    /**
     * Returns the metric dimensions.
     *
     * @return the metric dimensions
     */
    public Dimension[] getDimensions() {
        return dimensions;
    }

    /**
     * Returns a {@code Metric} instance parsed from the log string.
     *
     * @param line the input string
     * @return a {@code Metric} object
     */
    public static Metric parse(String line) {
        // DiskAvailable.Gigabytes:311|#Host:localhost|1650953744320
        Matcher matcher = PATTERN.matcher(line);
        if (!matcher.matches()) {
            return null;
        }

        String metricName = matcher.group(1);
        String unit = matcher.group(2);
        String value = matcher.group(3);
        String dimension = matcher.group(4);
        String timestamp = matcher.group(5);

        Dimension[] dimensions = null;
        if (dimension != null) {
            String[] dims = dimension.split(",");
            dimensions = new Dimension[dims.length];
            int index = 0;
            for (String dim : dims) {
                String[] pair = dim.split(":");
                if (pair.length == 2) {
                    dimensions[index++] = new Dimension(pair[0], pair[1]);
                }
            }
        }

        return new Metric(metricName, value, unit, timestamp, dimensions);
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(128);
        sb.append(metricName).append('.').append(unit).append(':').append(value);
        if (dimensions != null) {
            boolean first = true;
            for (Dimension dimension : dimensions) {
                if (dimension == null) {
                    continue;
                }
                if (first) {
                    sb.append("|#");
                    first = false;
                } else {
                    sb.append(',');
                }
                sb.append(dimension.getName()).append(':').append(dimension.getValue());
            }
        }
        if (timestamp != null) {
            sb.append('|').append(timestamp);
        }
        return sb.toString();
    }

    private static String getLocalHostName() {
        try {
            return InetAddress.getLocalHost().getHostName();
        } catch (UnknownHostException e) {
            return "Unknown";
        }
    }
}
