package org.tensorflow.engine;

import com.amazon.ai.Context;
import com.amazon.ai.Model;
import com.amazon.ai.TranslateException;
import com.amazon.ai.Translator;
import com.amazon.ai.TranslatorContext;
import com.amazon.ai.inference.Predictor;
import com.amazon.ai.metric.Metrics;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDFactory;
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.util.Pair;
import java.util.List;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class TfPredictor<I, O> implements Predictor<I, O> {

    TfNDFactory factory;
    Session session;
    Model model;
    private Translator<I, O> translator;
    List<String> outputNames;

    TfPredictor(TfModel model, Translator<I, O> translator, List<String> outputNames) {
        this.factory = TfNDFactory.SYSTEM_FACTORY.newSubFactory();
        this.translator = translator;
        this.session = model.getSession();
        this.model = model;
        this.outputNames = outputNames;
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
        Session.Runner runner = session.runner();
        for (Pair<String, NDArray> pair : ndList) {
            runner.feed(pair.getKey(), ((TfNDArray) pair.getValue()).getTensor()).run();
        }
        for (String outputName : outputNames) {
            runner.fetch(outputName);
        }
        List<Tensor<?>> result = runner.run();
        NDList resultNDList = new NDList();
        for (int i = 0; i < result.size(); i++) {
            resultNDList.add(outputNames.get(i), factory.create(result.get(i)));
        }

        return resultNDList;
    }

    public void setOutputNames(List<String> outputNames) {
        this.outputNames = outputNames;
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
            return null;
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
