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
import ai.djl.gluonTS.transform.GluonTSTransform;
import ai.djl.ndarray.NDManager;

import java.util.List;

/** Select preset field names. */
public class SelectField implements GluonTSTransform {

    private final List<String> inputFields;

    /**
     * Constructs a {@link SelectField}.
     *
     * @param inputFields field names to select from.
     */
    public SelectField(List<String> inputFields) {
        this.inputFields = inputFields;
    }

    /** {@inheritDoc} */
    @Override
    public GluonTSData transform(NDManager manager, GluonTSData data) {
        return Field.selectField(manager, inputFields, data);
    }
}
