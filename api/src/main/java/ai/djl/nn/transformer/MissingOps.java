/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.nn.transformer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;

/** Operators missing from NDArray that are necessary to implement Bert pretraining. */
public final class MissingOps {

    private MissingOps() {}

    /**
     * Applies the mxnet gather_nd operator.
     *
     * @param lookup array with the data to look up (e.g. an embedding table)
     * @param indices indices to use for lookup
     * @return the lookup result
     */
    public static NDArray gatherNd(NDArray lookup, NDArray indices) {
        return indices.getManager().invoke("gather_nd", new NDList(lookup, indices), null).head();
    }
}
