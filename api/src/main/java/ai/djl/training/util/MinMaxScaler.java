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
 * @author erik.bamberg@web.de
 */
public class MinMaxScaler {

    private NDArray fittedMin;
    private NDArray fittedMax;
    private NDArray fittedRange;
    private float minRange;
    private float maxRange = 1f;

    /**
     * compute the minimum and maximum to be used for later scaling.
     *
     * @param data used to compute the minimum and maximum used for later scaling
     * @param axises minimum maxium computation along this axises
     * @return the fitted MinMaxScaler
     */
    public MinMaxScaler fit(NDArray data, int[] axises) {
        fittedMin = data.min(axises);
        fittedMax = data.max(axises);
        fittedRange = fittedMax.sub(fittedMin);
        return this;
    }

    /**
     * compute the minimum and maximum to be used for later scaling.
     *
     * @param data used to compute the minimum and maximum used for later scaling
     * @return the fitted MinMaxScaler
     */
    public MinMaxScaler fit(NDArray data) {
        fit(data, new int[] {0});
        return this;
    }

    /**
     * transform the data using the previous calculated minimum and maximum.
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
        /*
        *  X *= self.scale_
              X += self.min_
        */
        NDArray std = data.sub(fittedMin).divi(fittedRange);
        return scale(std);
    }

    /**
     * transform the data in-place using the previous calculated minimum and maximum.
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
     * scale array from std if range is not default range otherwise return the unchanged array.
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
        } else {
            return std;
        }
    }

    /**
     * inverse scale array from std if range is not default range otherwise return the unchanged
     * array as a duplicate.
     *
     * @param std input array to scale
     * @return re-scaled array
     */
    private NDArray inverseScale(NDArray std) {
        // we don't have to scale by custom range when range is default 0..1
        if (maxRange != 1f || minRange != 0f) {
            return std.sub(minRange).divi(maxRange - minRange);
        } else {
            return std.duplicate();
        }
    }

    /**
     * inverse scale array from std if range is not default range otherwise return the array itself.
     *
     * <p>this is an inplace operation.
     *
     * @param std input array to scale in-place
     * @return re-scaled array
     */
    private NDArray inverseScalei(NDArray std) {
        // we don't have to scale by custom range when range is default 0..1
        if (maxRange != 1f || minRange != 0f) {
            return std.subi(minRange).divi(maxRange - minRange);
        } else {
            return std;
        }
    }

    /**
     * Undo the transformation of X according to feature_range.
     *
     * @param data to get transformed
     * @return the transformed array
     */
    public NDArray inverseTransform(NDArray data) {
        throwsIllegalStateWhenNotFitted();
        NDArray result = inverseScale(data);
        result = result.muli(fittedRange).addi(fittedMin);
        return result;
    }

    /**
     * Undo the transformation of X according to feature_range as an in-place operation.
     *
     * @param data to get transformed, the data get changed in-place
     * @return the transformed array
     */
    public NDArray inverseTransformi(NDArray data) {
        throwsIllegalStateWhenNotFitted();
        NDArray result = inverseScalei(data);
        result = result.muli(fittedRange).addi(fittedMin);
        return result;
    }

    /**
     * check if this MinMaxScaler is already fitted and throws exception otherwise.
     * throwsIllegalStateWhenNotFitted.
     */
    private void throwsIllegalStateWhenNotFitted() {
        if (fittedRange == null) {
            throw new IllegalStateException("Min Max Scaler is not fitted");
        }
    }

    /**
     * desired range of transformed data.
     *
     * @param minRange min value for desired range.
     * @param maxRange max value for desired range.
     * @return the configured MinMaxScaler.
     */
    public MinMaxScaler optRange(float minRange, float maxRange) {
        this.minRange = minRange;
        this.maxRange = maxRange;
        return this;
    }

    /**
     * gets the value of fittedMin.
     *
     * @return the fittedMin value.
     */
    public NDArray getMin() {
        throwsIllegalStateWhenNotFitted();
        return fittedMin;
    }

    /**
     * gets the value of fittedMax.
     *
     * @return the fittedMax value.
     */
    public NDArray getMax() {
        throwsIllegalStateWhenNotFitted();
        return fittedMax;
    }
}
