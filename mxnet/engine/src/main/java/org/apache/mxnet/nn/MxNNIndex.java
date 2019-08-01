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
package org.apache.mxnet.nn;

import org.apache.mxnet.nn.convolutional.MxConv1D;
import org.apache.mxnet.nn.convolutional.MxConv2D;
import org.apache.mxnet.nn.convolutional.MxConv3D;
import org.apache.mxnet.nn.core.MxLinear;
import org.apache.mxnet.nn.core.MxPrelu;
import org.apache.mxnet.nn.norm.MxBatchNorm;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.NNIndex;
import software.amazon.ai.nn.convolutional.Conv1D;
import software.amazon.ai.nn.convolutional.Conv2D;
import software.amazon.ai.nn.convolutional.Conv3D;
import software.amazon.ai.nn.core.Linear;
import software.amazon.ai.nn.core.Prelu;
import software.amazon.ai.nn.norm.BatchNorm;

public class MxNNIndex extends NNIndex {

    /** {@inheritDoc} */
    @Override
    public Linear linear(long outChannels, boolean bias) {
        return new MxLinear(outChannels, bias);
    }

    /** {@inheritDoc} */
    @Override
    public BatchNorm batchNorm2D(int axis, float epsilon, float momentum) {
        return new MxBatchNorm(axis, epsilon, momentum);
    }

    /** {@inheritDoc} */
    @Override
    public Prelu prelu() {
        return new MxPrelu();
    }

    /** {@inheritDoc} */
    @Override
    public Conv1D conv1D(
            Shape kernel,
            Shape stride,
            Shape pad,
            Shape dilate,
            int numFilters,
            int numGroups,
            boolean noBias) {
        return new MxConv1D(kernel, stride, pad, dilate, numFilters, numGroups, noBias);
    }

    /** {@inheritDoc} */
    @Override
    public Conv2D conv2D(
            Shape kernel,
            Shape stride,
            Shape pad,
            Shape dilate,
            int numFilters,
            int numGroups,
            boolean noBias) {
        return new MxConv2D(kernel, stride, pad, dilate, numFilters, numGroups, noBias);
    }

    /** {@inheritDoc} */
    @Override
    public Conv3D conv3D(
            Shape kernel,
            Shape stride,
            Shape pad,
            Shape dilate,
            int numFilters,
            int numGroups,
            boolean noBias) {
        return new MxConv3D(kernel, stride, pad, dilate, numFilters, numGroups, noBias);
    }
}
