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
import ai.djl.ndarray.NDList;
import ai.djl.translate.Translator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@code MxPredictor} is the MXNet implementation of {@link Predictor}.
 *
 * @param <I> the input object
 * @param <O> the output object
 * @see Predictor
 */
public class MxPredictor<I, O> extends BasePredictor<I, O> {

    private static final Logger logger = LoggerFactory.getLogger(MxPredictor.class);

    /**
     * Constructs a {@code MxPredictor}.
     *
     * @param model the model to predict with
     * @param translator the translator to convert with input and output
     * @param copy true if this is the first predictor created for the model (for thread safety)
     */
    MxPredictor(MxModel model, Translator<I, O> translator, boolean copy) {
        super(model, translator, copy);
    }

    /** {@inheritDoc} */
    @Override
    protected void waitToRead(NDList list) {
        list.forEach(a -> ((MxNDArray) a).waitToRead());
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
