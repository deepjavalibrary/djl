/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.examples.inference;

import ai.djl.ModelException;
import ai.djl.examples.inference.timeseries.AirPassengersDeepAR;
import ai.djl.examples.inference.timeseries.M5ForecastingDeepAR;
import ai.djl.testing.TestRequirements;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.util.Map;

public class TimeSeriesTest {

    private static final Logger logger = LoggerFactory.getLogger(TimeSeriesTest.class);

    @Test
    public void testM5Forecasting() throws ModelException, TranslateException, IOException {
        TestRequirements.linux();

        Map<String, Float> result = M5ForecastingDeepAR.predict();

        String[] metricNames =
                new String[] {
                    "RMSSE",
                    "MSE",
                    "abs_error",
                    "abs_target_sum",
                    "abs_target_mean",
                    "MAPE",
                    "sMAPE",
                    "ND"
                };
        for (String metricName : metricNames) {
            Assert.assertTrue(result.containsKey(metricName));
        }
    }

    @Test
    public void testAirPassenger() throws ModelException, TranslateException, IOException {
        TestRequirements.engine("MXNet");

        float[] result = AirPassengersDeepAR.predict();
        logger.info("{}", result);

        Assert.assertEquals(result.length, 12);
    }
}
