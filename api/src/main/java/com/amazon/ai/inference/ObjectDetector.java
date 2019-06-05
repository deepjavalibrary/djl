/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import com.amazon.ai.TranslateException;
import com.amazon.ai.Translator;
import com.amazon.ai.metric.Metrics;

public class ObjectDetector<I, O> implements AutoCloseable {

    private Predictor<I, O> predictor;

    public ObjectDetector(Model model, Translator<I, O> transformer) {
        this(model, transformer, Context.defaultContext());
    }

    public ObjectDetector(Model model, Translator<I, O> transformer, Context context) {
        this.predictor = Predictor.newInstance(model, transformer, context);
    }

    public O detect(I input) throws TranslateException {
        return predictor.predict(input);
    }

    public void setMetrics(Metrics metrics) {
        predictor.setMetrics(metrics);
    }

    @Override
    public void close() {
        predictor.close();
    }
}
