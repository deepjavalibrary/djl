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
package ai.djl.ml.xgboost;

import ai.djl.engine.Engine;
import ai.djl.engine.EngineProvider;

/** {@code XgbEngineProvider} is the XGBoost implementation of {@link EngineProvider}. */
public class XgbEngineProvider implements EngineProvider {

    private static Engine engine;

    /** {@inheritDoc} */
    @Override
    public String getEngineName() {
        return XgbEngine.ENGINE_NAME;
    }

    /** {@inheritDoc} */
    @Override
    public int getEngineRank() {
        return XgbEngine.RANK;
    }

    /** {@inheritDoc} */
    @Override
    public synchronized Engine getEngine() {
        if (engine == null) {
            engine = XgbEngine.newInstance();
        }
        return engine;
    }
}
