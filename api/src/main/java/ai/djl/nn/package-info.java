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

/**
 * Contains classes to construct neural networks.
 *
 * <p>The primary construct used to build up the networks is the {@link ai.djl.nn.Block} (see for
 * details). This package contains a number of implementations of blocks as well as helpers for
 * blocks.
 *
 * <p>The following subpackages also contain a number of standard neural network operations to use
 * with blocks:
 *
 * <ul>
 *   <li>{@link ai.djl.nn.convolutional}
 *   <li>{@link ai.djl.nn.core}
 *   <li>{@link ai.djl.nn.norm}
 *   <li>{@link ai.djl.nn.pooling}
 *   <li>{@link ai.djl.nn.recurrent}
 * </ul>
 */
package ai.djl.nn;
