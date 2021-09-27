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
package ai.djl.dlr.engine;

import ai.djl.Device;
import ai.djl.dlr.jni.JniUtils;
import ai.djl.inference.Predictor;
import ai.djl.translate.Translator;

/**
 * {@code DlrPredictor} is special implementation of {@link Predictor} for DLR.
 *
 * <p>The native Dlr doesn't support multi-threading feature, when creating a new DlrPredictor, we
 * copy the Dlr model handle to workaround the issue.
 */
public class DlrPredictor<I, O> extends Predictor<I, O> {
    /**
     * Creates a new instance of {@code DlrPredictor}.
     *
     * @param model the model on which the predictions are based
     * @param modelDir the path to the model artifacts
     * @param device the device that the model use
     * @param translator the translator to be used
     */
    public DlrPredictor(
            DlrModel model, String modelDir, Device device, Translator<I, O> translator) {
        super(model, translator, false);
        long modelHandle = JniUtils.createDlrModel(modelDir, device);
        block = new DlrSymbolBlock((DlrNDManager) manager, modelHandle);
        // disable cpu affinity by default
        JniUtils.useDlrCpuAffinity(modelHandle, false);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        super.close();
        ((DlrSymbolBlock) block).close();
    }
}
