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
package ai.djl.gluonTS.transform;

import ai.djl.gluonTS.GluonTSData;
import ai.djl.ndarray.NDManager;

/** This interface is used for data transformation on the {@link ai.djl.gluonTS.GluonTSData}. */
public interface GluonTSTransform {

    /**
     * Transform process on GluonTSData.
     *
     * @param manager The default manager for data process.
     * @param data The data to be operated on.
     * @return The result {@link GluonTSData}.
     */
    GluonTSData transform(NDManager manager, GluonTSData data);
}
