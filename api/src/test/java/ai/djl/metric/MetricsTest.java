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

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.List;
import java.util.Set;

public class MetricsTest {

    @Test
    public void testMetrics() {
        Metrics metrics = new Metrics();
        metrics.addMetric(new Metric("m1", 1L));
        metrics.addMetric("m1", 3L, Unit.COUNT);
        metrics.addMetric("m1", 2L);
        Metric p50 = metrics.percentile("m1", 50);
        Assert.assertEquals(p50.getValue().longValue(), 2L);

        metrics.addMetric(new Metric("m2", 1f));
        metrics.addMetric("m2", 3f, Unit.COUNT);
        metrics.addMetric("m2", 2f);
        p50 = metrics.percentile("m2", 50);
        Assert.assertEquals(p50.getValue().floatValue(), 2f);

        metrics.addMetric(new Metric("m3", 1d, Unit.NONE, new Dimension("model", "mlp")));
        metrics.addMetric("m3", 3d, Unit.COUNT);
        metrics.addMetric("m3", 2d);
        p50 = metrics.percentile("m3", 50);
        Assert.assertEquals(p50.getValue().doubleValue(), 2d);
        Assert.assertTrue(metrics.hasMetric("m1"));
        Assert.assertEquals(2L, metrics.latestMetric("m1").getValue().longValue());
        Assert.assertThrows(() -> metrics.latestMetric("none"));

        List<Metric> list = metrics.getMetric("m1");
        Assert.assertEquals(list.size(), 3);

        list = metrics.getMetric("m4");
        Assert.assertEquals(list.size(), 0);

        Set<String> metricNames = metrics.getMetricNames();
        Assert.assertEquals(metricNames.size(), 3);
        Assert.assertTrue(metricNames.contains("m1"));
        Assert.assertTrue(metricNames.contains("m2"));
        Assert.assertTrue(metricNames.contains("m3"));
        Assert.assertFalse(metricNames.contains("m4"));

        metrics.setLimit(3);
        metrics.setOnLimit(
                (m, n) -> {
                    Assert.assertEquals(m.getMetric(n).size(), 3);
                });
        metrics.addMetric("m1", 1L);
        Assert.assertEquals(metrics.getMetric("m1").size(), 1);
    }

    @Test
    public void testParseMetrics() {
        String line = "Disk.Gigabytes:311|#Host:localhost,Model:resnet|1650953744320|g";
        Metric metric = Metric.parse(line);
        Assert.assertNotNull(metric);
        Assert.assertEquals(metric.getValue().intValue(), 311);
        Assert.assertEquals(metric.getTimestamp(), "1650953744320");
        Assert.assertEquals(metric.getUnit(), Unit.GIGABYTES);
        Assert.assertEquals(metric.getDimensions()[0].getName(), "Host");
        Assert.assertEquals(metric.getDimensions()[0].getValue(), "localhost");
        Assert.assertEquals(line, metric.toString());
        Assert.assertEquals(metric.getMetricType(), MetricType.GAUGE);

        Assert.assertNull(Metric.parse("DiskAvailable.Gigabytes:311#"));
        Metric invalid = Metric.parse("Disk.InvalidUnits:311");
        Assert.assertNotNull(invalid);
        Assert.expectThrows(IllegalArgumentException.class, invalid::getUnit);
    }

    @Test
    public void testMetricsMean() {
        Metrics metrics = new Metrics();
        metrics.addMetric("m1", 2.4d);
        metrics.addMetric("m1", 3.4d);
        metrics.addMetric("m1", -1.3d);
        double mean = metrics.mean("m1");
        Assert.assertEquals(mean, 1.5d);
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testMeanException() {
        Metrics metrics = new Metrics();
        metrics.mean("not_found");
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testPercentileException() {
        Metrics metrics = new Metrics();
        metrics.percentile("not_found", 1);
    }

    @Test
    public void testMetricType() {
        Assert.assertEquals(MetricType.of("c"), MetricType.COUNTER);
        Assert.assertEquals(MetricType.of("g"), MetricType.GAUGE);
        Assert.assertEquals(MetricType.of("h"), MetricType.HISTOGRAM);
        Assert.assertThrows(() -> MetricType.of("a"));
    }
}
