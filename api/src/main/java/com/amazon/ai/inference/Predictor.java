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
import com.amazon.ai.Translator;
import com.amazon.ai.engine.Engine;
import com.amazon.ai.metric.Metrics;

public interface Predictor<I, O> extends AutoCloseable {

    static <I, O> Predictor<I, O> newInstance(Model model, Translator<I, O> transformer) {
        return newInstance(model, transformer, Context.defaultContext());
    }

    static <I, O> Predictor<I, O> newInstance(
            Model model, Translator<I, O> transformer, Context context) {
        return Engine.getInstance().newPredictor(model, transformer, context);
    }

    O predict(I input);

    void setMetrics(Metrics metrics);

    @Override
    void close();
}
