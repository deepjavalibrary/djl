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
 * Contains classes responsible for loading a deep learning engine.
 *
 * <p>Deep Java Library (DJL) is a higher level API that is used alongside implementations built on
 * other deep learning engines. By using only the higher level abstractions defined in the core DJL
 * API, it makes it easy to switch between underlying engines.
 *
 * <p>Each deep learning engine is implemented as a {@link java.util.ServiceLoader} and supplies an
 * implementation of the {@link ai.djl.engine.Engine} interface.
 *
 * @see ai.djl.engine.Engine
 * @see ai.djl.engine.EngineProvider
 */
package ai.djl.engine;
