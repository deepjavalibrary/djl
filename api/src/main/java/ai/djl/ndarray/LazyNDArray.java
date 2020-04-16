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
package ai.djl.ndarray;

/** An {@link NDArray} that waits to compute values until they are needed. */
public interface LazyNDArray extends NDArray {

    /** Runs the current NDArray and sleeps until the value is ready to read. */
    void waitToRead();

    /** Runs the current NDArray and sleeps until the value is ready to write. */
    void waitToWrite();

    /** Runs all NDArrays and sleeps until their values are fully computed. */
    void waitAll();
}
