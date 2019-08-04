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

import java.util.NoSuchElementException;
import java.util.Random;
import java.util.stream.IntStream;
import software.amazon.ai.util.RandomUtils;

public class RandomSampler implements Sampler {

    private int[] indices;
    private int pos;
    private Integer seed;

    public RandomSampler() {}

    public RandomSampler(int seed) {
        this.seed = seed;
    }

    /** {@inheritDoc} */
    @Override
    public final void init(int size) {
        pos = 0;
        indices = IntStream.range(0, size).toArray();
        if (seed == null) {
            for (int i = size; i > 1; --i) {
                swap(indices, i - 1, RandomUtils.nextInt(i));
            }
        } else {
            Random rnd = new Random(seed);
            for (int i = size; i > 1; --i) {
                swap(indices, i - 1, rnd.nextInt(i));
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasNext() {
        return pos < indices.length;
    }

    /** {@inheritDoc} */
    @Override
    public Integer next() {
        if (!hasNext()) {
            throw new NoSuchElementException();
        }
        return indices[pos++];
    }

    private static void swap(int[] arr, int i, int j) {
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}
