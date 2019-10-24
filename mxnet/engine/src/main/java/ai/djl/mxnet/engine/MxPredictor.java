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

import ai.djl.inference.BasePredictor;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Translator;
import ai.djl.util.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@code MxPredictor} is the MXNet implementation of {@link Predictor}.
 *
 * <p>MxPredictor contains all methods in the Predictor class and MXNet specific implementations.
 *
 * @param <I> Input Object
 * @param <O> Output Object
 */
public class MxPredictor<I, O> extends BasePredictor<I, O> {

    private static final Logger logger = LoggerFactory.getLogger(MxPredictor.class);

    MxPredictor(MxModel model, Translator<I, O> translator, boolean first) {
        super(model, translator, first);
    }

    @Override
    protected void waitToRead(NDList list) {
        for (Pair<String, NDArray> pair : list) {
            ((MxNDArray) pair.getValue()).waitToRead();
        }
    }

    /** {@inheritDoc} */
    @SuppressWarnings("deprecation")
    @Override
    protected void finalize() throws Throwable {
        if (((MxNDManager) manager).isOpen()) {
            if (logger.isDebugEnabled()) {
                logger.warn(
                        "MxPredictor was not closed explicitly: {}", getClass().getSimpleName());
            }
            close();
        }
        super.finalize();
    }
}
