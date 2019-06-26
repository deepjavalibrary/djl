package org.tensorflow.engine;

import software.amazon.ai.engine.Engine;
import software.amazon.ai.engine.EngineProvider;

public class TfEngineProvider implements EngineProvider {

    /** {@inheritDoc} */
    @Override
    public Engine getEngine() {
        return new TfEngine();
    }
}
