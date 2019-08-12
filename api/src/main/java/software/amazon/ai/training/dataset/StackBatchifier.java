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
package software.amazon.ai.training.dataset;

import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.ndarray.NDList;

/**
 * StackBatchifier is used to merge a list of samples to form a mini-batch of NDArray(s). The is
 * default Batchifier for data loading.
 */
public class StackBatchifier implements Batchifier {

    @Override
    public NDList batchify(NDList[] inputs) {
        // each input as NDList might contain several data or labels
        // so those should be batchified with counterpart
        int size = inputs[0].size();
        // if the NDList is empty
        if (size == 0) {
            return new NDList();
        }
        // collect all the data0...n in batch into one NDList
        NDList[] dataList = new NDList[size];
        for (NDList input : inputs) {
            for (int i = 0; i < size; i++) {
                if (dataList[i] == null) {
                    dataList[i] = new NDList();
                }
                dataList[i].add(input.get(i));
            }
        }
        // stack all the data and labels together
        NDList result = new NDList(size);
        for (NDList list : dataList) {
            result.add(NDArrays.stack(list));
        }

        return result;
    }
}
