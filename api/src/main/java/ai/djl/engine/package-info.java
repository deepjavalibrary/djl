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
 * Contains classes responsible for loading deep learning framework.
 *
 * <p>djl.ai API is deep learning framework agnostic. This package defines abstraction to hide the
 * difference between each framework.
 *
 * <p>Each deep learning framework is implemented as a service provider and supply an implementation
 * of {@link ai.djl.engine.Engine} interface.
 *
 * @see ai.djl.engine.Engine
 * @see ai.djl.engine.EngineProvider
 */
package ai.djl.engine;
