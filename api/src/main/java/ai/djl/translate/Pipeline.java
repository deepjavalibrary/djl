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
import ai.djl.util.PairList;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class Pipeline {

    private PairList<String, Transform> transforms;

    public Pipeline() {
        transforms = new PairList<>();
    }

    public Pipeline(Transform... transforms) {
        this.transforms = new PairList<>();
        for (Transform transform : transforms) {
            this.transforms.add(null, transform);
        }
    }

    public Pipeline add(Transform transform) {
        transforms.add(null, transform);
        return this;
    }

    public Pipeline add(int index, Transform transform) {
        transforms.add(index, null, transform);
        return this;
    }

    public Pipeline add(String name, Transform transform) {
        transforms.add(name, transform);
        return this;
    }

    public Pipeline add(int index, String name, Transform transform) {
        transforms.add(index, name, transform);
        return this;
    }

    public NDList transform(NDList input) {
        if (transforms.isEmpty() || input.isEmpty()) {
            return input;
        }

        NDArray[] arrays = input.toArray(new NDArray[0]);

        Map<String, Integer> map = new ConcurrentHashMap<>();
        // create mapping
        for (int i = 0; i < input.size(); i++) {
            String key = input.get(i).getName();
            if (key != null) {
                map.put(key, i);
            }
        }
        // apply transform
        for (Pair<String, Transform> transform : transforms) {
            String key = transform.getKey();
            int index;
            if (key != null) {
                index = map.get(key);
            } else {
                index = 0;
            }
            NDArray array = arrays[index];

            arrays[index] = transform.getValue().transform(array);
            arrays[index].setName(array.getName());
        }

        return new NDList(arrays);
    }
}
