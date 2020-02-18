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
package ai.djl.tensorflow.engine;

import ai.djl.inference.BasePredictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.util.List;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class TfPredictor<I, O> extends BasePredictor<I, O> {

    public TfPredictor(TfModel model, Translator<I, O> translator, boolean copy) {
        super(model, translator, copy);
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forward(TranslatorContext ctx, NDList ndList) {
        Session session = ((TfModel) model).getSession();
        Session.Runner runner = session.runner();
        for (NDArray array : ndList) {
            runner.feed("serving_default_input_1:0", ((TfNDArray) array).getTensor());
        }
        runner.fetch("StatefulPartitionedCall:0");
        List<Tensor<?>> result = runner.run();

        NDList resultNDList = new NDList();
        for (Tensor<?> tensor : result) {
            resultNDList.add(((TfNDManager) model.getNDManager()).create(tensor));
        }

        return resultNDList;
    }
}
