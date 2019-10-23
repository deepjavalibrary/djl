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

import ai.djl.Device;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;

/** Batch is used to get a batch of data and labels from {@link Dataset}. */
public class Batch implements AutoCloseable {

    private NDManager manager;
    private NDList data;
    private NDList labels;
    private Batchifier batchifier;

    public Batch(NDManager manager, NDList data, NDList labels) {
        this.manager = manager;
        data.attach(manager);
        labels.attach(manager);
        this.data = data;
        this.labels = labels;
    }

    public Batch(NDManager manager, NDList data, NDList labels, Batchifier batchifier) {
        this.manager = manager;
        data.attach(manager);
        labels.attach(manager);
        this.data = data;
        this.labels = labels;
        this.batchifier = batchifier;
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

    public Batch[] split(Device[] devices, boolean evenSplit) {
        int size = devices.length;
        if (size == 1) {
            NDList d = data.asInDevice(devices[0], true);
            NDList l = labels.asInDevice(devices[0], true);
            return new Batch[] {new Batch(manager, d, l, batchifier)};
        }

        NDList[] splittedData = split(data, size, evenSplit);
        NDList[] splittedLabels = split(labels, size, evenSplit);

        Batch[] splitted = new Batch[splittedData.length];
        for (int i = 0; i < splittedData.length; ++i) {
            NDList d = splittedData[i].asInDevice(devices[i], true);
            NDList l = splittedLabels[i].asInDevice(devices[i], true);
            splitted[i] = new Batch(manager, d, l, batchifier);
        }
        return splitted;
    }

    private NDList[] split(NDList list, int numOfSlices, boolean evenSplit) {
        if (batchifier == null) {
            throw new IllegalStateException(
                    "Split can only be called on a batch containing a batchifier");
        }
        return batchifier.split(list, numOfSlices, evenSplit);
    }
}
