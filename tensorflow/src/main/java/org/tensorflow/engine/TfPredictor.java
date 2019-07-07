package org.tensorflow.engine;

import java.util.List;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import software.amazon.ai.Context;
import software.amazon.ai.Model;
import software.amazon.ai.TranslateException;
import software.amazon.ai.Translator;
import software.amazon.ai.TranslatorContext;
import software.amazon.ai.inference.Predictor;
import software.amazon.ai.metric.Metrics;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDFactory;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.util.Pair;

public class TfPredictor<I, O> implements Predictor<I, O> {

    TfNDFactory factory;
    Session session;
    Model model;
    private Translator<I, O> translator;

    public TfPredictor(TfModel model, Translator<I, O> translator) {
        this.factory = TfNDFactory.SYSTEM_FACTORY.newSubFactory();
        this.translator = translator;
        this.session = model.getSession();
        this.model = model;
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("PMD.AvoidRethrowingException")
    public O predict(I input) throws TranslateException {
        try (PredictorContext inputCtx = new PredictorContext();
                PredictorContext outputCtx = new PredictorContext()) {
            NDList ndList = translator.processInput(inputCtx, input);

            NDList result = forward(ndList, model);

            return translator.processOutput(outputCtx, result);
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new TranslateException(e);
        }
    }

    private NDList forward(NDList ndList, Model model) {
        Session.Runner runner = session.runner();
        for (Pair<String, NDArray> pair : ndList) {
            runner.feed(pair.getKey(), ((TfNDArray) pair.getValue()).getTensor());
        }
        // TODO We can extract input name from decribeInput in Model if NDList doesn't have names
        DataDesc[] dataDescs = model.describeOutput();
        for (DataDesc desc : dataDescs) {
            runner.fetch(desc.getName());
        }
        List<Tensor<?>> result = runner.run();

        NDList resultNDList = new NDList();
        for (int i = 0; i < result.size(); i++) {
            resultNDList.add(dataDescs[i].getName(), factory.create(result.get(i)));
        }

        return resultNDList;
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
