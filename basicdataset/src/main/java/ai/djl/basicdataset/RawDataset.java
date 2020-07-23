/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.basicdataset;

import ai.djl.training.dataset.Dataset;
import java.io.IOException;

/**
 * An interface can read a plain java object from dataset.
 *
 * @param <T> the raw data type of the dataset
 */
public interface RawDataset<T> extends Dataset {

    /**
     * Returns a plain java object.
     *
     * @return a plain java object
     * @throws IOException when IO operation fails in loading a resource
     */
    T getData() throws IOException;
}
