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
 * Contains classes for evaluating the effectiveness of models.
 *
 * <p>It contains a main interface {@link ai.djl.training.evaluator.Evaluator} and various
 * evaluators that extend it. More evaluators are located within {@link ai.djl.training.loss} which
 * have the additional property that those evaluators are suited for training.
 */
package ai.djl.training.evaluator;
