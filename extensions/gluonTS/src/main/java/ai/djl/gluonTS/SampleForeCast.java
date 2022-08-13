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
import ai.djl.ndarray.NDManager;

import java.time.LocalDateTime;

/**
 * A {@code Forecast} object, where the predicted distribution is represented internally as samples.
 */
public class SampleForeCast extends ForeCast {

    protected static NDManager samplesManger = NDManager.newBaseManager();

    private NDArray samples;
    private String item_id;
    private int numSamples;

    private NDArray sortedSamples = null;

    /**
     * Constructs a {@link SampleForeCast}.
     *
     * @param samples {@link NDArray} array of size (num_samples, prediction_length) (1D case),
     *     (num_samples, prediction_length, target_dim) (multivariate case)
     * @param startDate start of the forecast
     * @param itemId id
     * @param freq frequency of the forecast.
     */
    public SampleForeCast(NDArray samples, LocalDateTime startDate, String itemId, String freq) {
        super(startDate, (int) samples.getShape().get(1), freq);
        this.samples = samplesManger.create(samples.getShape());
        samples.copyTo(this.samples);
        this.item_id = itemId;
        this.numSamples = (int) samples.getShape().head();
    }

    /**
     * Returns the sorted sample array.
     *
     * @return the sorted sample array.
     */
    public NDArray getSortedSamples() {
        if (sortedSamples == null) {
            sortedSamples = samples.sort(0);
        }
        return sortedSamples;
    }

    /**
     * Returns the number of samples representing the forecast.
     *
     * @return the number of samples.
     */
    public int getNumSamples() {
        return numSamples;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray quantile(float q) {
        int sampleIdx = Math.round((numSamples - 1) * q);
        return getSortedSamples().get("{}, :", sampleIdx);
    }

    /**
     * Returns a new Forecast object with only the selected sub-dimension.
     *
     * @param dim the selected dim.
     * @return a new {@link SampleForeCast}.
     */
    public SampleForeCast copyDIm(int dim) {
        NDArray samples;
        if (this.samples.getShape().dimension() == 2) {
            samples = this.samples;
        } else {
            int targetDim = (int) this.samples.getShape().get(2);
            if (dim >= targetDim) {
                throw new IllegalArgumentException(
                        String.format(
                                "must set 0 <= dim < target_dim, but got dim=%d, target_dim=%d",
                                dim, targetDim));
            }
            samples = this.samples.get(":, :, {}", dim);
        }

        return new SampleForeCast(samples, startDate, item_id, freq);
    }

    @Override
    public NDArray mean() {
        if (mean == null) {
            mean = samples.mean(new int[] {0});
        }
        return super.mean();
    }
}
