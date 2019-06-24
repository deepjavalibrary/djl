package org.tensorflow.engine;

import com.amazon.ai.TranslateException;
import com.amazon.ai.inference.Predictor;
import com.amazon.ai.metric.Metrics;

public class TfPredictor<I, O> implements Predictor<I, O> {

    /** {@inheritDoc} */
    @Override
    public O predict(I input) throws TranslateException {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void setMetrics(Metrics metrics) {}

    /** {@inheritDoc} */
    @Override
    public void close() {}
}
