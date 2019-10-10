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
import java.util.stream.IntStream;

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

    public Pipeline add(String name, Transform transform) {
        transforms.add(name, transform);
        return this;
    }

    public NDList transform(NDList input, boolean close) {
        if (transforms.isEmpty() || input.isEmpty()) {
            return input;
        }

        NDArray[] arrays = input.toArray();
        String[] strings = new String[input.size()];

        Map<String, Integer> map = new ConcurrentHashMap<>();
        // create mapping
        for (int i = 0; i < input.size(); i++) {
            String key = input.getWithTag(i).getKey();
            if (key != null) {
                map.put(key, i);
            }
            strings[i] = key;
        }
        // apply transform
        for (Pair<String, Transform> transform : transforms) {
            int index = (transform.getKey() != null) ? map.getOrDefault(transform.getKey(), -1) : 0;
            if (index == -1) {
                throw new IllegalArgumentException(
                        transform.getKey() + " can't be found in input NDList");
            }
            arrays[index] = transform.getValue().transform(arrays[index], close);
        }
        // restore the NDList
        NDList res = new NDList(input.size());
        IntStream.range(0, input.size()).forEach(i -> res.add(strings[i], arrays[i]));
        return res;
    }
}
