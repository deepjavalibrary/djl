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
package ai.djl.training.dataset;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;

/** Batch is used to get a batch of data and labels from {@link Dataset}. */
public class Batch implements AutoCloseable {

    private NDManager manager;
    private NDList data;
    private NDList labels;

    public Batch(NDManager manager, NDList data, NDList labels) {
        this.manager = manager;
        data.attach(manager);
        labels.attach(manager);
        this.data = data;
        this.labels = labels;
    }

    public NDManager getManager() {
        return manager;
    }

    public NDList getData() {
        return data;
    }

    public NDList getLabels() {
        return labels;
    }

    @Override
    public void close() {
        manager.close();
        manager = null;
    }
}
