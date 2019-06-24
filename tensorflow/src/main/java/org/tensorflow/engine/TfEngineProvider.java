package org.tensorflow.engine;

import com.amazon.ai.engine.Engine;
import com.amazon.ai.engine.EngineProvider;

public class TfEngineProvider implements EngineProvider {

    /** {@inheritDoc} */
    @Override
    public Engine getEngine() {
        return new TfEngine();
    }
}
