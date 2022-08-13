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
package ai.djl.gluonTS.dataset;

/** Represents the name field of elements in a {@link ai.djl.gluonTS.GluonTSData} as an enum. */
public enum FieldName {
    ITEM_ID,

    START,
    TARGET,

    FEAT_STATIC_CAT,
    FEAT_STATIC_REAL,
    FEAT_DYNAMIC_CAT,
    FEAT_DYNAMIC_REAL,
    PAST_FEAT_DYNAMIC_REAL,
    FEAT_DYNAMIC_REAL_LEGACY,

    FEAT_DYNAMIC,
    PAST_FEAT_DYNAMIC,

    FEAT_TIME,
    FEAT_CONST,
    FEAT_AGE,

    OBSERVED_VALUES,
    IS_PAD,
    FORECAST_START,

    TARGET_DIM_INDICATOR;

    /**
     * Converts the name to lower case.
     *
     * @return the String converted to lower case.
     */
    public String lowerCase() {
        return this.toString().toLowerCase();
    }
}
