/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.repository.dataset;

import ai.djl.util.Progress;
import java.io.IOException;

/**
 * A {@code PreparedDataset} is a {@link ai.djl.training.dataset.Dataset} that requires an
 * additional preparation step before use.
 *
 * <p>The preparation steps can be run by calling {@link PreparedDataset#prepare()}.
 */
public interface PreparedDataset {

    /**
     * Prepares the dataset for use.
     *
     * @throws IOException for various exceptions depending on the dataset
     */
    default void prepare() throws IOException {
        prepare(null);
    }

    /**
     * Prepares the dataset for use with tracked progress.
     *
     * @param progress the progress tracker
     * @throws IOException for various exceptions depending on the dataset
     */
    void prepare(Progress progress) throws IOException;
}
