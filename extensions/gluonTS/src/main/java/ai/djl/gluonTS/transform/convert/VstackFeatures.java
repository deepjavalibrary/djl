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
package ai.djl.gluonTS.transform.convert;

import ai.djl.gluonTS.GluonTSData;
import ai.djl.gluonTS.dataset.FieldName;
import ai.djl.gluonTS.transform.GluonTSTransform;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;

import java.util.List;

/** Use the {@link ai.djl.ndarray.NDArrays#concat(NDList)} to vstack datas. */
public class VstackFeatures implements GluonTSTransform {

    private FieldName outputField;
    private List<FieldName> inputFields;
    private boolean dropInputs;
    private boolean hStack;

    /**
     * Constructs a {@link VstackFeatures}.
     *
     * @param outputField output field name
     * @param inputFields input field names
     * @param dropInputs whether to remove the input fields
     * @param hStack use hStack
     */
    public VstackFeatures(
            FieldName outputField,
            List<FieldName> inputFields,
            boolean dropInputs,
            boolean hStack) {
        this.outputField = outputField;
        this.inputFields = inputFields;
        this.dropInputs = dropInputs;
        this.hStack = hStack;
    }

    /**
     * Constructs a {@link VstackFeatures}.
     *
     * @param outputField output field name
     * @param inputFields input field names
     */
    public VstackFeatures(FieldName outputField, List<FieldName> inputFields) {
        this(outputField, inputFields, true, false);
    }

    /** {@inheritDoc} */
    @Override
    public GluonTSData transform(NDManager manager, GluonTSData data) {
        return Convert.vstackFeatures(manager, outputField, inputFields, dropInputs, hStack, data);
    }
}
