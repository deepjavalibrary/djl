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

    private PairList<IndexKey, Transform> transforms;

    public Pipeline() {
        transforms = new PairList<>();
    }

    public Pipeline(Transform... transforms) {
        this.transforms = new PairList<>();
        for (Transform transform : transforms) {
            this.transforms.add(new IndexKey(0), transform);
        }
    }

    public Pipeline add(Transform transform) {
        transforms.add(new IndexKey(0), transform);
        return this;
    }

    public Pipeline add(int index, Transform transform) {
        transforms.add(index, new IndexKey(0), transform);
        return this;
    }

    public Pipeline add(String name, Transform transform) {
        transforms.add(new IndexKey(name), transform);
        return this;
    }

    public Pipeline insert(int position, Transform transform) {
        transforms.add(position, new IndexKey(0), transform);
        return this;
    }

    public Pipeline insert(int position, int index, Transform transform) {
        transforms.add(position, new IndexKey(index), transform);
        return this;
    }

    public Pipeline insert(int position, String name, Transform transform) {
        transforms.add(position, new IndexKey(name), transform);
        return this;
    }

    public NDList transform(NDList input) {
        if (transforms.isEmpty() || input.isEmpty()) {
            return input;
        }

        NDArray[] arrays = input.toArray(new NDArray[0]);

        Map<IndexKey, Integer> map = new ConcurrentHashMap<>();
        // create mapping
        for (int i = 0; i < input.size(); i++) {
            String key = input.get(i).getName();
            if (key != null) {
                map.put(new IndexKey(key), i);
            }
            map.put(new IndexKey(i), i);
        }
        // apply transform
        for (Pair<IndexKey, Transform> transform : transforms) {
            IndexKey key = transform.getKey();
            int index = map.get(key);
            NDArray array = arrays[index];

            arrays[index] = transform.getValue().transform(array);
            arrays[index].setName(array.getName());
        }

        return new NDList(arrays);
    }

    private static final class IndexKey {
        private String key;
        private int index;

        private IndexKey(String key) {
            this.key = key;
        }

        private IndexKey(int index) {
            this.index = index;
        }

        @Override
        public int hashCode() {
            if (key == null) {
                return index;
            }
            return key.hashCode();
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) {
                return true;
            }
            if (!(obj instanceof IndexKey)) {
                return false;
            }
            IndexKey other = (IndexKey) obj;
            if (key == null) {
                return index == other.index;
            }
            return key.equals(other.key);
        }
    }
}
