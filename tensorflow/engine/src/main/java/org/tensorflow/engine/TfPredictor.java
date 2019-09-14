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
package org.tensorflow.engine;

import java.util.List;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import software.amazon.ai.Device;
import software.amazon.ai.Model;
import software.amazon.ai.inference.Predictor;
import software.amazon.ai.metric.Metrics;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.translate.TranslateException;
import software.amazon.ai.translate.Translator;
import software.amazon.ai.translate.TranslatorContext;
import software.amazon.ai.util.Pair;

public class TfPredictor<I, O> implements Predictor<I, O> {

    TfNDManager manager;
    Session session;
    Model model;
    private Translator<I, O> translator;

    public TfPredictor(TfModel model, Translator<I, O> translator) {
        this.manager = TfNDManager.newBaseManager();
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
            resultNDList.add(dataDescs[i].getName(), manager.create(result.get(i)));
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
        private TfNDManager ctxManager;

        public PredictorContext() {
            ctxManager = manager.newSubManager();
        }
        /** {@inheritDoc} */
        @Override
        public Model getModel() {
            return model;
        }

        /** {@inheritDoc} */
        @Override
        public Device getDevice() {
            return null;
        }

        /** {@inheritDoc} */
        @Override
        public NDManager getNDManager() {
            return ctxManager;
        }

        /** {@inheritDoc} */
        @Override
        public Metrics getMetrics() {
            return null;
        }

        /** {@inheritDoc} */
        @Override
        public void close() {
            ctxManager.close();
        }
    }
}
