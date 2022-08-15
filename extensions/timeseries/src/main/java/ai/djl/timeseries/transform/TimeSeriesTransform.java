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
import ai.djl.timeseries.TimeSeriesData;

/** This interface is used for data transformation on the {@link TimeSeriesData}. */
public interface TimeSeriesTransform {

    /**
     * Transform process on TimeSeriesData.
     *
     * @param manager The default manager for data process
     * @param data The data to be operated on
     * @return The result {@link TimeSeriesData}.
     */
    TimeSeriesData transform(NDManager manager, TimeSeriesData data);
}
