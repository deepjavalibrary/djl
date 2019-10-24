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
import ai.djl.ndarray.types.DataDesc;
import ai.djl.translate.Translator;
import ai.djl.util.Pair;
import java.util.List;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class TfPredictor<I, O> extends BasePredictor<I, O> {

    public TfPredictor(TfModel model, Translator<I, O> translator, boolean first) {
        super(model, translator, first);
    }

    @Override
    protected NDList forward(NDList ndList) {
        Session session = ((TfNDManager) model).getSession();
        TfNDManager tfNDManager = (TfNDManager) manager;
        Session.Runner runner = session.runner();
        for (Pair<String, NDArray> pair : ndList) {
            runner.feed(pair.getKey(), ((TfNDArray) pair.getValue()).getTensor());
        }
        // TODO We can extract input name from decribeInput in Model if NDList doesn't have names
        DataDesc[] dataDescs = model.describeOutput();
        for (DataDesc desc : dataDescs) {
            runner.fetch(desc.getName());
        }
        List<Tensor<?>> result = runner.run();

        NDList resultNDList = new NDList();
        for (int i = 0; i < result.size(); i++) {
            resultNDList.add(dataDescs[i].getName(), tfNDManager.create(result.get(i)));
        }

        return resultNDList;
    }
}
