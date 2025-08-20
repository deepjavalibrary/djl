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
package ai.djl.mxnet.engine;

import ai.djl.engine.Engine;
import ai.djl.engine.EngineProvider;

/** {@code MxEngineProvider} is the MXNet implementation of {@link EngineProvider}. */
public class MxEngineProvider implements EngineProvider {

    /** {@inheritDoc} */
    @Override
    public String getEngineName() {
        return MxEngine.ENGINE_NAME;
    }

    /** {@inheritDoc} */
    @Override
    public int getEngineRank() {
        String osName = System.getProperty("os.name");
        String osArch = System.getProperty("os.arch");
        if (osName.startsWith("Mac") && "aarch64".equals(osArch)) {
            // MXNet doesn't support macOS M1
            return 99;
        }
        return MxEngine.RANK;
    }

    /** {@inheritDoc} */
    @Override
    public Engine getEngine() {
        return InstanceHolder.INSTANCE;
    }

    private static final class InstanceHolder {
        static final Engine INSTANCE = MxEngine.newInstance();
    }
}
