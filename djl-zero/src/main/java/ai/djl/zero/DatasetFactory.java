/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.zero;

import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.Dataset.Usage;

/**
 * A Factory to help construct datasets for the different {@link Usage}s and provide metadata about
 * the datasets.
 */
public interface DatasetFactory {

    /**
     * Builds the dataset for the given {@link Usage}.
     *
     * @param usage what part of the training process the dataset will be used for
     * @return the dataset for the given {@link Usage}
     */
    Dataset build(Usage usage);
}
