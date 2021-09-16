/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.tensorrt.engine;

import ai.djl.inference.Predictor;
import ai.djl.translate.Translator;

class TrtPredictor<I, O> extends Predictor<I, O> {

    TrtPredictor(TrtModel model, Translator<I, O> translator, TrtSession session) {
        super(model, translator, false);
        block = session;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        super.close();
        ((TrtSession) block).close();
    }
}
