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
package org.apache.mxnet.engine;

import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDManager;

/** Images utility functions that contains image read, decode and translation. */
public final class MxImages {

    private MxImages() {}

    public static NDArray read(NDManager manager, String path) {
        return read(manager, path, Flag.COLOR);
    }

    public static NDArray read(NDManager manager, String path, Flag flag) {
        return ((MxNDManager) manager).imread(path, flag.ordinal());
    }

    public enum Flag {
        GRAYSCALE,
        COLOR
    }
}
