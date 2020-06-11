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
package ai.djl.ndarray.index.dim;

import ai.djl.ndarray.NDArray;

/** An {@code NDIndexElement} to return values based on a mask binary NDArray. */
public class NDIndexBooleans implements NDIndexElement {

    private NDArray index;

    /**
     * Constructs a {@code NDIndexBooleans} instance with specified mask binary NDArray.
     *
     * @param index the mask binary {@code NDArray}
     */
    public NDIndexBooleans(NDArray index) {
        this.index = index;
    }

    /**
     * Returns the mask binary {@code NDArray}.
     *
     * @return the mask binary {@code NDArray}
     */
    public NDArray getIndex() {
        return index;
    }

    /** {@inheritDoc} */
    @Override
    public int getRank() {
        return index.getShape().dimension();
    }
}
