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
package software.amazon.ai.nn;

import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.convolutional.Conv1D;
import software.amazon.ai.nn.convolutional.Conv2D;
import software.amazon.ai.nn.convolutional.Conv3D;
import software.amazon.ai.nn.core.Linear;
import software.amazon.ai.nn.norm.BatchNorm;

/**
 * An internal mapping to Engine specific implementations of Neural Network {@link
 * software.amazon.ai.Block}s.
 */
public abstract class NNIndex {

    public abstract Linear linear(long units, boolean bias);

    public abstract BatchNorm batchNorm2D(int axis, float epsilon, float momentum);

    public abstract Conv1D conv1D(
            Shape kernel,
            Shape stride,
            Shape pad,
            Shape dilate,
            int numFilters,
            int numGroups,
            boolean noBias);

    public abstract Conv2D conv2D(
            Shape kernel,
            Shape stride,
            Shape pad,
            Shape dilate,
            int numFilters,
            int numGroups,
            boolean noBias);

    public abstract Conv3D conv3D(
            Shape kernel,
            Shape stride,
            Shape pad,
            Shape dilate,
            int numFilters,
            int numGroups,
            boolean noBias);
}
