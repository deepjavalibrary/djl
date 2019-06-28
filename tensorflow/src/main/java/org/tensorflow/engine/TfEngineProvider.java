package org.tensorflow.engine;

import software.amazon.ai.engine.Engine;
import software.amazon.ai.engine.EngineProvider;

public class TfEngineProvider implements EngineProvider {

    private static final Engine ENGINE = new TfEngine();

    /** {@inheritDoc} */
    @Override
    public Engine getEngine() {
        return ENGINE;
    }
}
