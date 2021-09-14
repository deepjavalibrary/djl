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

package ai.djl.tflite.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.nn.AbstractSymbolBlock;
import ai.djl.nn.SymbolBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import org.tensorflow.lite.Interpreter;

/**
 * {@code TfLiteSymbolBlock} is the TFLite implementation of {@link SymbolBlock}.
 *
 * <p>You can create a {@code TfLiteSymbolBlock} using {@link ai.djl.Model#load(java.nio.file.Path,
 * String)}.
 */
public class TfLiteSymbolBlock extends AbstractSymbolBlock implements AutoCloseable {

    private TfLiteNDManager manager;
    private Interpreter interpreter;

    TfLiteSymbolBlock(Interpreter interpreter, TfLiteNDManager manager) {
        this.interpreter = interpreter;
        this.manager = manager;
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        Object[] intInput = inputs.stream().map(NDArray::toByteBuffer).toArray();
        interpreter.runForMultipleInputsOutputs(intInput);

        int outputSize = interpreter.getOutputTensorCount();
        NDList result = new NDList(outputSize);
        for (int i = 0; i < outputSize; i++) {
            result.add(manager.createInternal(interpreter.getOutputTensor(i)));
        }
        result.attach(inputs.head().getManager());
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        interpreter.close();
    }
}
