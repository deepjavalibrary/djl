/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.timeseries.model.deepar;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/** A deepar implements for training. */
public final class DeepARTrainingNetwork extends DeepARNetwork {

    DeepARTrainingNetwork(Builder builder) {
        super(builder);
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArray featStaticCat = inputs.get(0);
        NDArray featStaticReal = inputs.get(1);
        NDArray pastTimeFeat = inputs.get(2);
        NDArray pastTarget = inputs.get(3);
        NDArray pastObservedValues = inputs.get(4);
        //        NDArray pastIsPad = inputs.get(5);
        NDArray futureTimeFeat = inputs.get(6);
        NDArray futureTarget = inputs.get(7);
        NDArray futureObservedValues = inputs.get(8);

        NDList unrollOutput =
                unrollLaggedRnn(
                        parameterStore,
                        new NDList(
                                featStaticCat,
                                featStaticReal,
                                pastTimeFeat,
                                pastTarget,
                                pastObservedValues,
                                futureTimeFeat,
                                futureTarget),
                        training);

        NDArray observedValues =
                pastObservedValues
                        .get(":, {}:", -contextLength + 1)
                        .concat(futureObservedValues, 1);
        observedValues.setName("loss_weights");

        String[] argNames = distrOutput.getArgsArray();
        NDList ret = new NDList(argNames.length + 2); // args + scale + loss_weights

        for (String argName : argNames) {
            ret.add(unrollOutput.get(argName));
        }
        ret.add(unrollOutput.get("scale"));
        ret.add(observedValues);
        return ret;
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        Shape targetShape = inputShapes[3].slice(2);
        Shape contextShape = new Shape(1, contextLength).addAll(targetShape);
        Shape scaleShape = scaler.getOutputShapes(new Shape[] {contextShape, contextShape})[1];
        long scaleSize = scaleShape.get(1);

        long embeddedCatSize = embedder.getOutputShapes(new Shape[] {inputShapes[0]})[0].get(1);

        Shape inputShape = new Shape(1, contextLength * 2L - 1).addAll(targetShape);
        Shape lagsShape = inputShape.add(lagsSeq.size());
        long featSize = inputShapes[2].get(2) + embeddedCatSize + inputShapes[1].get(1) + scaleSize;
        Shape rnnInputShape =
                lagsShape.slice(0, lagsShape.dimension() - 1).add(lagsShape.tail() + featSize);

        Shape rnnOutShape = rnn.getOutputShapes(new Shape[] {rnnInputShape})[0];
        Shape[] argShapes = paramProj.getOutputShapes(new Shape[] {rnnOutShape});

        long[] observedValueShape = new long[inputShapes[8].dimension()];
        System.arraycopy(
                inputShapes[8].getShape(), 0, observedValueShape, 0, observedValueShape.length);
        observedValueShape[1] += contextLength - 1;
        Shape lossWeightsShape = new Shape(observedValueShape);

        Shape[] ret = new Shape[argShapes.length + 2];
        System.arraycopy(argShapes, 0, ret, 0, argShapes.length);
        ret[argShapes.length] = scaleShape;
        ret[argShapes.length + 1] = lossWeightsShape;
        return ret;
    }
}
