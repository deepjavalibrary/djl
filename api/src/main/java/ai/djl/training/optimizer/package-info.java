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
 * Contains classes for optimizing a neural network {@link ai.djl.nn.Block}.
 *
 * <p>It contains a main interface {@link ai.djl.training.optimizer.Optimizer} and various
 * optimizers that extend it. There are also the helpers for learning rates in {@link
 * ai.djl.training.tracker}.
 */
package ai.djl.training.optimizer;
