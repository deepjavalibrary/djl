/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.training.util;

import ai.djl.ndarray.NDArray;

/**
 * Transform arrays by scaling each value to a given range. The desired range of transformed data
 * can be set using {@code optRange}. the range defaults to 0...1.
 *
 * <p>After fitting the scaler the fitted values are attached to the same NDManager as the input
 * array.
 *
 * @author erik.bamberg@web.de
 */
public class MinMaxScaler implements AutoCloseable {

    private NDArray fittedMin;
    private NDArray fittedMax;
    private NDArray fittedRange;
    private float minRange;
    private float maxRange = 1f;
    private boolean detached;

    /**
     * Computes the minimum and maximum to be used for later scaling.
     *
     * <p>After fitting the scaler the fitted values are attached to the same NDManager as the input
     * array. reusing the minMaxScaler in the context of other NDManager's is possible by {@code
     * detach()} the scaler from the NDManager.
     *
     * @param data used to compute the minimum and maximum used for later scaling
     * @param axises minimum maximum computation along this axises
     * @return the fitted MinMaxScaler
     */
    public MinMaxScaler fit(NDArray data, int[] axises) {
        fittedMin = data.min(axises);
        fittedMax = data.max(axises);
        fittedRange = fittedMax.sub(fittedMin);
        if (detached) {
            detach();
        }
        return this;
    }

    /**
     * Computes the minimum and maximum to be used for later scaling.
     *
     * <p>After fitting the scaler the fitted values are attached to the same NDManager as the input
     * array. reusing the minMaxScaler in the context of other NDManager's is possible by {@code
     * detach()} the scaler from the NDManager.
     *
     * @param data used to compute the minimum and maximum used for later scaling
     * @return the fitted MinMaxScaler
     */
    public MinMaxScaler fit(NDArray data) {
        fit(data, new int[] {0});
        return this;
    }

    /**
     * Transforms the data using the previous calculated minimum and maximum.
     *
     * <p>if {@code fit()} is not executed yet, then the minimum/maximum is computer based on the
     * input data array and used for later computations. X_std = (X - X.min(axis=0)) /
     * (X.max(axis=0) - X.min(axis=0)) X_scaled = X_std * (max - min) + min
     *
     * @param data to get transformed
     * @return the transformed data, the input array is not changed
     */
    public NDArray transform(NDArray data) {
        if (fittedRange == null) {
            fit(data, new int[] {0});
        }
        NDArray std = data.sub(fittedMin).divi(fittedRange);
        return scale(std);
    }

    /**
     * Transforms the data in-place using the previous calculated minimum and maximum.
     *
     * <p>if {@code fit()} is not called before then the minimum/maximum is computer based on the
     * input data array and used for later computations. X_std = (X - X.min(axis=0)) /
     * (X.max(axis=0) - X.min(axis=0)) X_scaled = X_std * (max - min) + min
     *
     * @param data to get transformed
     * @return the transformed data (reference to the input data) the input array is changed
     *     in-place by this operation
     */
    public NDArray transformi(NDArray data) {
        if (fittedRange == null) {
            fit(data, new int[] {0});
        }
        NDArray std = data.subi(fittedMin).divi(fittedRange);
        return scale(std);
    }

    /**
     * Scales array from std if range is not default range otherwise return the unchanged array.
     *
     * <p>this is an in-place operation.
     *
     * @param std input array to scale
     * @return scaled array
     */
    private NDArray scale(NDArray std) {
        // we don't have to scale by custom range when range is default 0..1
        if (maxRange != 1f || minRange != 0f) {
            return std.muli(maxRange - minRange).addi(minRange);
        }
        return std;
    }

    /**
     * Inverses scale array from std if range is not default range otherwise return the unchanged
     * array as a duplicate.
     *
     * @param std input array to scale
     * @return re-scaled array
     */
    private NDArray inverseScale(NDArray std) {
        // we don't have to scale by custom range when range is default 0..1
        if (maxRange != 1f || minRange != 0f) {
            return std.sub(minRange).divi(maxRange - minRange);
        }
        return std.duplicate();
    }

    /**
     * Inverses scale array from std if range is not default range otherwise return the array
     * itself.
     *
     * <p>this is an in-place operation.
     *
     * @param std input array to scale in-place
     * @return re-scaled array
     */
    private NDArray inverseScalei(NDArray std) {
        // we don't have to scale by custom range when range is default 0..1
        if (maxRange != 1f || minRange != 0f) {
            return std.subi(minRange).divi(maxRange - minRange);
        }
        return std;
    }

    /**
     * Undoes the transformation of X according to feature_range.
     *
     * @param data to get transformed
     * @return the transformed array
     */
    public NDArray inverseTransform(NDArray data) {
        throwsIllegalStateWhenNotFitted();
        NDArray result = inverseScale(data);
        return result.muli(fittedRange).addi(fittedMin);
    }

    /**
     * Undoes the transformation of X according to feature_range as an in-place operation.
     *
     * @param data to get transformed, the data get changed in-place
     * @return the transformed array
     */
    public NDArray inverseTransformi(NDArray data) {
        throwsIllegalStateWhenNotFitted();
        NDArray result = inverseScalei(data);
        return result.muli(fittedRange).addi(fittedMin);
    }

    /**
     * Checks if this MinMaxScaler is already fitted and throws exception otherwise.
     *
     * @throws IllegalStateException when not Fitted
     */
    private void throwsIllegalStateWhenNotFitted() {
        if (fittedRange == null) {
            throw new IllegalStateException("Min Max Scaler is not fitted");
        }
    }

    /**
     * Detaches this MinMaxScaler fitted value from current NDManager's lifecycle.
     *
     * <p>this becomes un-managed and it is the user's responsibility to close this. Failure to
     * close the resource might cause your machine to run out of native memory.
     *
     * <p>After fitting the scaler the fitted values are attached to the same NDManager as the input
     * array.
     *
     * <p>Re-fitting the scaler after detaching doesn't re-attach the scaler to any NDManager.
     *
     * @return the detached MinMaxScaler (itself) - to use as a fluent API
     */
    public MinMaxScaler detach() {
        detached = true;
        if (fittedMin != null) {
            fittedMin.detach();
        }
        if (fittedMax != null) {
            fittedMax.detach();
        }
        if (fittedRange != null) {
            fittedRange.detach();
        }
        return this;
    }

    /**
     * Sets desired range of transformed data.
     *
     * @param minRange min value for desired range
     * @param maxRange max value for desired range
     * @return the configured MinMaxScaler
     */
    public MinMaxScaler optRange(float minRange, float maxRange) {
        this.minRange = minRange;
        this.maxRange = maxRange;
        return this;
    }

    /**
     * Returns the value of fittedMin.
     *
     * @return the fittedMin value
     */
    public NDArray getMin() {
        throwsIllegalStateWhenNotFitted();
        return fittedMin;
    }

    /**
     * Returns the value of fittedMax.
     *
     * @return the fittedMax value
     */
    public NDArray getMax() {
        throwsIllegalStateWhenNotFitted();
        return fittedMax;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (fittedMin != null) {
            fittedMin.close();
        }
        if (fittedMax != null) {
            fittedMax.close();
        }
        if (fittedRange != null) {
            fittedRange.close();
        }
    }
}
