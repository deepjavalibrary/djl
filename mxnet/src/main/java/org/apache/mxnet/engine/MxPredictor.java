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
package org.apache.mxnet.engine;

import com.amazon.ai.Context;
import com.amazon.ai.Transformer;
import com.amazon.ai.inference.Predictor;
import com.amazon.ai.ndarray.NDList;

public class MxPredictor<I, O> implements Predictor<I, O>, AutoCloseable {

    private MxModel model;
    private Transformer<I, O> transformer;
    private Context context;
    private Module module;
    private ResourceAllocator alloc;

    MxPredictor(MxModel model, Transformer<I, O> transformer, Context context) {
        this.model = model;
        this.transformer = transformer;
        this.context = context;
        alloc = new ResourceAllocator();
        Module.Builder builder = new Module.Builder(context, model, false);
        module = builder.build(alloc);
    }

    @Override
    public O predict(I input) {
        try (NDList ndList = transformer.processInput(input);
                NDList result = forward(ndList)) {
            return transformer.processOutput(result);
        }
    }

    public MxModel getModel() {
        return model;
    }

    public Context getContext() {
        return context;
    }

    private NDList forward(NDList ndList) {
        rebindIfNeeded(ndList);

        return module.forward(ndList);
    }

    private void rebindIfNeeded(NDList ndList) {}

    @Override
    public void close() {
        alloc.close();
    }
}
