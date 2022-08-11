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
package ai.djl.gluonTS;

import ai.djl.ndarray.NDArray;

import java.time.LocalDateTime;

/** An abstract class representing the forecast results for a time series in a GluonTS case. */
public abstract class ForeCast {

    protected LocalDateTime startDate;
    protected int predictionLength;
    protected NDArray mean = null;
    protected String freq;

    /**
     * Constructs a Forecast, during the post-processing.
     *
     * @param startDate the time series start date.
     * @param predictionLength the time length of prediction.
     * @param freq the prediction frequency
     */
    public ForeCast(LocalDateTime startDate, int predictionLength, String freq) {
        this.startDate = startDate;
        this.predictionLength = predictionLength;
        this.freq = freq;
    }

    /**
     * Computes a quantile from the predicted distribution.
     *
     * @param q Quantile to compute
     * @return value of the quantile across the prediction range.
     */
    public abstract NDArray quantile(float q);

    /**
     * Computes a quantile from the predicted distribution.
     *
     * @param q Quantile to compute
     * @return value of the quantile across the prediction range.
     */
    public NDArray quantile(String q) {
        return this.quantile(Float.parseFloat(q));
    }

    public NDArray median() {
        return this.quantile(0.5f);
    }

    public String freq() {
        return this.freq;
    }

    /**
     * Returns the time length of forecast.
     *
     * @return
     */
    public int predictionLength() {
        return predictionLength;
    }

    /**
     * Returns forecast mean
     *
     * @return
     */
    public NDArray mean() {
        return mean;
    }

    public void plot() {
        throw new UnsupportedOperationException("plot is not supported yet");
    }
}
