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
package ai.djl.timeseries.transform.field;

import ai.djl.ndarray.NDManager;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.timeseries.transform.TimeSeriesTransform;

import java.util.List;

/** Remove the field names. */
public class RemoveFields implements TimeSeriesTransform {

    private final List<FieldName> fieldNames;

    /**
     * Constructs a {@code RemoveFields} instance.
     *
     * @param fieldNames field name to be removed
     */
    public RemoveFields(List<FieldName> fieldNames) {
        this.fieldNames = fieldNames;
    }

    /** {@inheritDoc} */
    @Override
    public TimeSeriesData transform(NDManager manager, TimeSeriesData data, boolean isTrain) {
        Field.removeFields(fieldNames, data);
        return data;
    }
}
