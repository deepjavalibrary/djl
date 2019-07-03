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
package software.amazon.ai.engine;

/**
 * The <code>EngineProvider</code> instance manufactures an {@link Engine} instance, which is
 * available in the system.
 *
 * <p>At initialization time, {@link java.util.ServiceLoader} will search <code>EngineProvider
 * </code> implementations available in the class path.
 *
 * <p>Engine is designed as a singleton. {@link Engine#getInstance()} will only return the first
 * Engine found in the class path. However, you can directly create a specific Engine instance (e.g.
 * <code>
 * MxEngine</code>).
 */
public interface EngineProvider {

    /**
     * Returns the instance of {@link Engine} class that EngineProvider should bind to.
     *
     * @return the instance of {@link Engine}
     */
    Engine getEngine();
}
