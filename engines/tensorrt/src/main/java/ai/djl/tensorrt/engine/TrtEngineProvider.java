/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.tensorrt.engine;

import ai.djl.engine.Engine;
import ai.djl.engine.EngineProvider;

/** {@code TrtEngineProvider} is the TensorRT implementation of {@link EngineProvider}. */
public class TrtEngineProvider implements EngineProvider {

    private static volatile Engine engine; // NOPMD

    /** {@inheritDoc} */
    @Override
    public String getEngineName() {
        return TrtEngine.ENGINE_NAME;
    }

    /** {@inheritDoc} */
    @Override
    public int getEngineRank() {
        return TrtEngine.RANK;
    }

    /** {@inheritDoc} */
    @Override
    public Engine getEngine() {
        if (engine == null) {
            synchronized (this) {
                engine = TrtEngine.newInstance();
            }
        }
        return engine;
    }
}
