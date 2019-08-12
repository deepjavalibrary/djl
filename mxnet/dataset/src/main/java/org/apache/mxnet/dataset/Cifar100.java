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
package org.apache.mxnet.dataset;

import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.training.dataset.Dataset;
import software.amazon.ai.util.Pair;

public class Cifar100 implements Dataset {

    @Override
    public Iterable<Pair<NDList, NDList>> getData() {
        return null;
    }
}
