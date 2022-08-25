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
package ai.djl.timeseries;

import ai.djl.ndarray.NDArray;

import java.time.LocalDateTime;

/** An abstract class representing the forecast results for the time series data. */
public abstract class Forecast {

    protected LocalDateTime startDate;
    protected int predictionLength;
    protected String freq;

    /**
     * Constructs a {@code Forecast} instance.
     *
     * @param startDate the time series start date
     * @param predictionLength the time length of prediction
     * @param freq the prediction frequency
     */
    public Forecast(LocalDateTime startDate, int predictionLength, String freq) {
        this.startDate = startDate;
        this.predictionLength = predictionLength;
        this.freq = freq;
    }

    /**
     * Computes a quantile from the predicted distribution.
     *
     * @param q quantile to compute
     * @return value of the quantile across the prediction range
     */
    public abstract NDArray quantile(float q);

    /**
     * Computes a quantile from the predicted distribution.
     *
     * @param q quantile to compute
     * @return value of the quantile across the prediction range
     */
    public NDArray quantile(String q) {
        return quantile(Float.parseFloat(q));
    }

    /**
     * Computes and returns the forecast mean.
     *
     * @return forecast mean
     */
    public abstract NDArray mean();

    /**
     * Computes the median of forecast.
     *
     * @return value of the median
     */
    public NDArray median() {
        return quantile(0.5f);
    }

    /**
     * Returns the prediction frequency like "D", "H"....
     *
     * @return the prediction frequency
     */
    public String freq() {
        return freq;
    }

    /**
     * Returns the time length of forecast.
     *
     * @return the prediction length
     */
    public int getPredictionLength() {
        return predictionLength;
    }
}
