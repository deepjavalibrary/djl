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

/** Record is used to get a single element of data and labels from {@link Dataset}. */
public class Record {

    private NDList data;
    private NDList labels;

    public Record(NDList data, NDList labels) {
        this.data = data;
        this.labels = labels;
    }

    public NDList getData() {
        return data;
    }

    public NDList getLabels() {
        return labels;
    }
}
