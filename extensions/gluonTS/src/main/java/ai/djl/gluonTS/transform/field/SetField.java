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
package ai.djl.gluonTS.transform.field;

import ai.djl.gluonTS.GluonTSData;
import ai.djl.gluonTS.dataset.FieldName;
import ai.djl.gluonTS.transform.GluonTSTransform;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/** Use the preset value for input field names. */
public class SetField implements GluonTSTransform {

    private final FieldName outputField;
    private final NDArray value;

    /**
     * Constructs a {@link SetField}.
     *
     * @param outputField output field name to be set.
     * @param value value to be set.
     */
    public SetField(FieldName outputField, NDArray value) {
        this.outputField = outputField;
        this.value = value;
    }

    /** {@inheritDoc} */
    @Override
    public GluonTSData transform(NDManager manager, GluonTSData data) {
        NDArray copyValue = manager.create(value.getShape());
        value.copyTo(copyValue);
        return Field.setField(manager, outputField, copyValue, data);
    }
}
