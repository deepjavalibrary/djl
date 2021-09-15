/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.paddlepaddle.engine;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.translate.Translator;

/**
 * {@code PpPredictor} is special implementation of {@link Predictor} for PaddlePaddle.
 *
 * <p>When creating a new DlrPredictor, we clone Paddle predictor handle to workaround the issue.
 */
public class PpPredictor<I, O> extends Predictor<I, O> {

    PaddlePredictor predictor;

    /**
     * Creates a new instance of {@code PaddlePredictor}.
     *
     * @param model the model on which the predictions are based
     * @param predictor the C++ Paddle Predictor handle
     * @param translator the translator to be used
     */
    public PpPredictor(Model model, PaddlePredictor predictor, Translator<I, O> translator) {
        super(model, translator, false);
        this.predictor = predictor;
        block = new PpSymbolBlock(predictor, (PpNDManager) model.getNDManager());
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        super.close();
        this.predictor.close();
    }
}
