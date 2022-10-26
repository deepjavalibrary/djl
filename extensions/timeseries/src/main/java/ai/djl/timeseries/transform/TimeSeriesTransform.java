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
package ai.djl.timeseries.transform;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;

import java.util.ArrayList;
import java.util.List;

/** This interface is used for data transformation on the {@link TimeSeriesData}. */
public interface TimeSeriesTransform {

    /**
     * Transform process on TimeSeriesData.
     *
     * @param manager The default manager for data process
     * @param data The data to be operated on
     * @param isTrain Whether it is training
     * @return The result {@link TimeSeriesData}.
     */
    TimeSeriesData transform(NDManager manager, TimeSeriesData data, boolean isTrain);

    /**
     * Construct a list of {@link TimeSeriesTransform} that performs identity function.
     *
     * @return a list of identity {@link TimeSeriesTransform}
     */
    static List<TimeSeriesTransform> identityTransformation() {
        List<TimeSeriesTransform> ret = new ArrayList<>();
        ret.add(new IdentityTransform());
        return ret;
    }

    /** An identity transformation. */
    class IdentityTransform implements TimeSeriesTransform {

        /** {@inheritDoc} */
        @Override
        public TimeSeriesData transform(NDManager manager, TimeSeriesData data, boolean isTrain) {
            data.setField("PAST_" + FieldName.TARGET, data.get(FieldName.TARGET));
            data.setField("FUTURE_" + FieldName.TARGET, manager.create(new Shape(0)));
            return data;
        }
    }
}
