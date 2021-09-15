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
package ai.djl.paddlepaddle.engine;

import ai.djl.paddlepaddle.jni.JniUtils;
import ai.djl.util.NativeResource;

/** PaddlePaddle C++ Predictor. */
public class PaddlePredictor extends NativeResource<Long> {

    PaddlePredictor(long handle) {
        super(handle);
    }

    /** {@inheritDoc} */
    public PaddlePredictor copy() {
        return new PaddlePredictor(JniUtils.clonePredictor(this));
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        JniUtils.deletePredictor(this);
    }
}
