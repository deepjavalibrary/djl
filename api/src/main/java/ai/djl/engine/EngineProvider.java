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
package ai.djl.engine;

/**
 * The {@code EngineProvider} instance manufactures an {@link Engine} instance, which is available
 * in the system.
 *
 * <p>At initialization time, the {@link java.util.ServiceLoader} will search for {@code
 * EngineProvider} implementations available in the class path.
 *
 * <p>{@link Engine} is designed as a collection of singletons. {@link Engine#getInstance()} will
 * return the default Engine, which is the first one found in the classpath. Many of the standard
 * APIs will rely on this default Engine instance such as when creating a {@link
 * ai.djl.ndarray.NDManager} or {@link ai.djl.Model}. However, you can directly get a specific
 * Engine instance (e.g. {@code MxEngine}) by calling {@link Engine#getEngine(String)}.
 */
public interface EngineProvider {

    /**
     * Returns the name of the {@link Engine}.
     *
     * @return the name of {@link Engine}
     */
    String getEngineName();

    /**
     * Returns the rank of the {@link Engine}.
     *
     * @return the rank of {@link Engine}
     */
    int getEngineRank();

    /**
     * Returns the instance of the {@link Engine} class EngineProvider should bind to.
     *
     * @return the instance of {@link Engine}
     */
    Engine getEngine();
}
