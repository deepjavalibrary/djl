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
package ai.djl.translate;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.util.Pair;

public interface Transform {

    default NDArray transform(NDArray array) {
        return transform(array, false);
    }

    NDArray transform(NDArray array, boolean close);

    default NDList transform(NDList list) {
        return transform(list, false);
    }

    default NDList transform(NDList list, boolean close) {
        NDList result = new NDList(list.size());
        for (Pair<String, NDArray> pair : list) {
            result.add(pair.getKey(), transform(pair.getValue(), close));
        }
        return result;
    }
}
