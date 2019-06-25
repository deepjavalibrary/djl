package org.tensorflow.engine;

import com.amazon.ai.Context;
import com.amazon.ai.Model;
import com.amazon.ai.TranslateException;
import com.amazon.ai.Translator;
import com.amazon.ai.TranslatorContext;
import com.amazon.ai.inference.Predictor;
import com.amazon.ai.metric.Metrics;
import com.amazon.ai.ndarray.NDFactory;
import com.amazon.ai.ndarray.NDList;
import org.tensorflow.Session;

public class TfPredictor<I, O> implements Predictor<I, O> {

    TfNDFactory factory;
    Session session;
    Model model;
    Context context;
    private Translator<I, O> translator;

    TfPredictor(TfModel model, Translator<I, O> translator, Context context) {
        this.factory = TfNDFactory.SYSTEM_FACTORY.newSubFactory(context);
        this.translator = translator;
        this.session = model.getSession();
        this.model = model;
        this.context = context;
    }


    /** {@inheritDoc} */
    @Override
    public O predict(I input) throws TranslateException {
        PredictorContext inputCtx = new PredictorContext();
        PredictorContext outputCtx = new PredictorContext();

        NDList ndList = translator.processInput(inputCtx, input);

        NDList result = forward(ndList);

        return translator.processOutput(outputCtx, result);
    }

    private NDList forward(NDList ndList) {
        
    }

    /** {@inheritDoc} */
    @Override
    public void setMetrics(Metrics metrics) {}

    /** {@inheritDoc} */
    @Override
    public void close() {}

    private class PredictorContext implements TranslatorContext {
        private TfNDFactory ctxFactory;

        public PredictorContext() {
            ctxFactory = factory.newSubFactory();
        }
            /** {@inheritDoc} */
            @Override
            public Model getModel() {
                return model;
            }

            /** {@inheritDoc} */
            @Override
            public Context getContext() {
                return context;
            }

            /** {@inheritDoc} */
            @Override
            public NDFactory getNDFactory() {
                return ctxFactory;
            }

            /** {@inheritDoc} */
            @Override
            public Metrics getMetrics() {
                return null;
            }

            /** {@inheritDoc} */
            @Override
            public void close() {
                ctxFactory.close();
            }
    }
}
