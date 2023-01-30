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
package ai.djl.basicdataset.tabular.utils;

/** An interface that convert String to numeric data. */
public interface Featurizer {

    /**
     * Puts encoded data into the float buffer.
     *
     * @param buf the float buffer to be filled
     * @param input the string input
     */
    void featurize(DynamicBuffer buf, String input);

    /**
     * Returns the length of the data array required by {@link #deFeaturize(float[])}.
     *
     * @return the length of the data array required by {@link #deFeaturize(float[])}
     */
    int dataRequired();

    /**
     * Converts the output data for a label back into the Java type.
     *
     * @param data the data vector correspondign to the feature
     * @return a Java type (depending on the {@link Featurizer}) representing the data.
     */
    Object deFeaturize(float[] data);

    /**
     * A {@link Featurizer} that only supports the data featurize operations, but not the full
     * deFeaturize operations used by labels.
     */
    interface DataFeaturizer extends Featurizer {

        /** {@inheritDoc} */
        @Override
        default int dataRequired() {
            throw new IllegalStateException(
                    "DataFeaturizers only support featurize, not deFeaturize");
        }

        /** {@inheritDoc} */
        @Override
        default Object deFeaturize(float[] data) {
            throw new IllegalStateException(
                    "DataFeaturizers only support featurize, not deFeaturize");
        }
    }
}
