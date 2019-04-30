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
package com.amazon.ai.inference;

import com.amazon.ai.Context;
import com.amazon.ai.Model;
import com.amazon.ai.Transformer;
import com.amazon.ai.engine.Engine;
import com.amazon.ai.ndarray.NDArray;

public class Predictor<I, O> {

    protected Predictor<NDArray, NDArray> predictor;
    protected Transformer<I, O> transformer;

    public Predictor(Model model, Transformer<I, O> transformer) {
        this(model, transformer, Context.defaultContext());
    }

    public Predictor(Model model, Transformer<I, O> transformer, Context context) {
        predictor = Engine.getInstance().newPredictor(model, context);
        this.transformer = transformer;
    }

    protected Predictor() {}

    public O predict(I input) {
        try (NDArray array = transformer.processInput(input);
                NDArray result = predictor.predict(array)) {
            return transformer.processOutput(result);
        }
    }
}
