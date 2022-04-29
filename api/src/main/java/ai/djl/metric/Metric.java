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
import java.util.ArrayList;
import java.util.List;
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
                    "\\s*([\\w\\s]+)\\.([\\w\\s]+):([0-9\\-,.e]+)\\|#([^|]*)\\|#hostname:([^,]+),([^,]+)(,.*)?");

    private static final String LOCALHOST = getLocalHostName();

    @SerializedName("MetricName")
    private String metricName;

    @SerializedName("Value")
    private String value;

    @SerializedName("Unit")
    private String unit;

    @SerializedName("Dimensions")
    private List<Dimension> dimensions;

    @SerializedName("Timestamp")
    private String timestamp;

    @SerializedName("HostName")
    private String hostName;

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
        this.metricName = metricName;
        this.value = value.toString();
        this.unit = unit.getValue();
        this.hostName = LOCALHOST;
    }

    /**
     * Constructs a new {@code Metric} instance.
     *
     * @param metricName the metric name
     * @param unit the metric unit
     * @param value the metric value
     * @param hostName the host name
     * @param timestamp the metric timestamp
     */
    private Metric(
            String metricName, String unit, String value, String hostName, String timestamp) {
        this.metricName = metricName;
        this.unit = unit;
        this.value = value;
        this.hostName = hostName;
        this.timestamp = timestamp;
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
     * Returns the timestamp of the {@code Metric}.
     *
     * @return the metric timestamp
     */
    public String getHostName() {
        return hostName;
    }

    /**
     * Returns a {@code Metric} instance parsed from the log string.
     *
     * @param line the input string
     * @return a {@code Metric} object
     */
    public static Metric parse(String line) {
        // DiskAvailable.Gigabytes:311|#Level:Host|#hostname:localhost,1650953744320,request_id
        Matcher matcher = PATTERN.matcher(line);
        if (!matcher.matches()) {
            return null;
        }

        Metric metric =
                new Metric(
                        matcher.group(1),
                        matcher.group(2),
                        matcher.group(3),
                        matcher.group(5),
                        matcher.group(6));

        String dimensions = matcher.group(4);
        if (dimensions != null) {
            String[] dimension = dimensions.split(",");
            List<Dimension> list = new ArrayList<>(dimension.length);
            for (String dime : dimension) {
                String[] pair = dime.split(":");
                if (pair.length == 2) {
                    list.add(new Dimension(pair[0], pair[1]));
                }
            }
            metric.dimensions = list;
        }

        return metric;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(128);
        sb.append(metricName).append('.').append(unit).append(':').append(value).append("|#");
        if (dimensions != null) {
            boolean first = true;
            for (Dimension dimension : dimensions) {
                if (first) {
                    first = false;
                } else {
                    sb.append(',');
                }
                sb.append(dimension.getName()).append(':').append(dimension.getValue());
            }
        }
        sb.append("|#hostname:").append(hostName);
        sb.append(',').append(timestamp);
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
