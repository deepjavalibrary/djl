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

import ai.djl.ndarray.NDList;
import java.util.Arrays;
import java.util.stream.IntStream;

public interface Batchifier {

    Batchifier STACK = new StackBatchifier();

    NDList batchify(NDList[] inputs);

    NDList[] unbatchify(NDList inputs);

    default NDList[] split(NDList list, int numOfSlices, boolean evenSplit) {
        NDList[] unbatched = unbatchify(list);
        int batchSize = unbatched.length;
        numOfSlices = Math.min(numOfSlices, batchSize);
        if (evenSplit && batchSize % numOfSlices != 0) {
            throw new IllegalArgumentException(
                    "data with shape "
                            + batchSize
                            + " cannot be evenly split into "
                            + numOfSlices
                            + ". Use a batch size that's multiple of "
                            + numOfSlices
                            + " or set even_split=true to allow"
                            + " uneven partitioning of data.");
        }

        NDList[] splitted = new NDList[numOfSlices];
        Arrays.setAll(splitted, i -> new NDList());

        int step = (int) Math.ceil((double) batchSize / numOfSlices);
        for (int i = 0; i < numOfSlices; i++) {
            NDList[] currentUnbatched =
                    IntStream.range(i * step, Math.min((i + 1) * step, batchSize))
                            .mapToObj(j -> unbatched[j])
                            .toArray(NDList[]::new);
            splitted[i] = batchify(currentUnbatched);
        }
        return splitted;
    }
}
