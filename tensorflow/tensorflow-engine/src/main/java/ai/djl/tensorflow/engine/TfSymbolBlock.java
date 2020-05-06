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

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.BlockList;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterList;
import ai.djl.nn.SymbolBlock;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.Initializer;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.List;
import java.util.Map;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.proto.framework.MetaGraphDef;
import org.tensorflow.proto.framework.SignatureDef;
import org.tensorflow.proto.framework.TensorInfo;
import org.tensorflow.proto.framework.TensorShapeProto;

public class TfSymbolBlock implements SymbolBlock {

    private NDManager manager;
    private SavedModelBundle bundle;
    private MetaGraphDef metaGraphDef;
    private Session session;

    public TfSymbolBlock(NDManager manager, SavedModelBundle bundle) {
        this.manager = manager;
        this.bundle = bundle;
        session = bundle.session();
        metaGraphDef = bundle.metaGraphDef();
    }

    /** {@inheritDoc} */
    @Override
    public void removeLastBlock() {
        throw new UnsupportedOperationException("Not supported for TensorFlow Engine");
    }

    /** {@inheritDoc} */
    @Override
    public NDList forward(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        Session.Runner runner = session.runner();
        PairList<String, Shape> inputDescriptions = describeInput();
        PairList<String, Shape> outputDescriptions = describeOutput();

        for (int i = 0; i < inputDescriptions.size(); i++) {
            runner.feed(inputDescriptions.get(i).getKey(), ((TfNDArray) inputs.get(i)).getTensor());
        }
        for (int i = 0; i < outputDescriptions.size(); i++) {
            runner.fetch(outputDescriptions.get(i).getKey());
        }
        List<Tensor<?>> result = runner.run();

        NDList resultNDList = new NDList();
        for (Tensor<?> tensor : result) {
            resultNDList.add(((TfNDManager) manager).create(tensor));
        }
        return resultNDList;
    }

    /** {@inheritDoc} */
    @Override
    public void setInitializer(Initializer initializer) {
        throw new UnsupportedOperationException("Not supported for TensorFlow Engine");
    }

    /** {@inheritDoc} */
    @Override
    public void setInitializer(Initializer initializer, String paramName) {
        throw new UnsupportedOperationException("Not supported for TensorFlow Engine");
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] initialize(NDManager manager, DataType dataType, Shape... inputShapes) {
        return new Shape[0];
    }

    /** {@inheritDoc} */
    @Override
    public boolean isInitialized() {
        return bundle != null;
    }

    /** {@inheritDoc} */
    @Override
    public void cast(DataType dataType) {
        throw new UnsupportedOperationException("Not supported for TensorFlow Engine");
    }

    /** {@inheritDoc} */
    @Override
    public void clear() {
        if (session != null) {
            session.close();
        }
        if (bundle != null) {
            bundle.close();
        }
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeInput() {
        PairList<String, Shape> inputDescriptions = new PairList<>();
        Map<String, SignatureDef> signatureDefMap = metaGraphDef.getSignatureDefMap();
        SignatureDef servingDefault =
                metaGraphDef.getSignatureDefOrDefault(
                        "serving_default",
                        signatureDefMap.get(signatureDefMap.keySet().iterator().next()));
        for (Map.Entry<String, TensorInfo> entry : servingDefault.getInputsMap().entrySet()) {
            TensorShapeProto shapeProto = entry.getValue().getTensorShape();
            inputDescriptions.add(
                    entry.getValue().getName(),
                    new Shape(
                            shapeProto
                                    .getDimList()
                                    .stream()
                                    .mapToLong(TensorShapeProto.Dim::getSize)
                                    .toArray()));
        }
        return inputDescriptions;
    }

    PairList<String, Shape> describeOutput() {
        PairList<String, Shape> outputDescription = new PairList<>();
        Map<String, SignatureDef> signatureDefMap = metaGraphDef.getSignatureDefMap();
        SignatureDef servingDefault =
                metaGraphDef.getSignatureDefOrDefault(
                        "serving_default",
                        signatureDefMap.get(signatureDefMap.keySet().iterator().next()));
        for (Map.Entry<String, TensorInfo> entry : servingDefault.getOutputsMap().entrySet()) {
            TensorShapeProto shapeProto = entry.getValue().getTensorShape();
            outputDescription.add(
                    entry.getValue().getName(),
                    new Shape(
                            shapeProto
                                    .getDimList()
                                    .stream()
                                    .mapToLong(TensorShapeProto.Dim::getSize)
                                    .toArray()));
        }
        return outputDescription;
    }

    /** {@inheritDoc} */
    @Override
    public BlockList getChildren() {
        throw new UnsupportedOperationException("Not supported for TensorFlow Engine");
    }

    /** {@inheritDoc} */
    @Override
    public List<Parameter> getDirectParameters() {
        throw new UnsupportedOperationException("Not supported for TensorFlow Engine");
    }

    /** {@inheritDoc} */
    @Override
    public ParameterList getParameters() {
        throw new UnsupportedOperationException("Not supported for TensorFlow Engine");
    }

    /** {@inheritDoc} */
    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        throw new UnsupportedOperationException("Not supported for TensorFlow Engine");
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes) {
        return new Shape[0];
    }

    /** {@inheritDoc} */
    @Override
    public void saveParameters(DataOutputStream os) {
        throw new UnsupportedOperationException("Not supported for TensorFlow Engine");
    }

    /** {@inheritDoc} */
    @Override
    public void loadParameters(NDManager manager, DataInputStream is) {
        throw new UnsupportedOperationException("Not supported for TensorFlow Engine");
    }
}
