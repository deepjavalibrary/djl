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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractSymbolBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.proto.framework.MetaGraphDef;
import org.tensorflow.proto.framework.SignatureDef;
import org.tensorflow.proto.framework.TensorInfo;
import org.tensorflow.proto.framework.TensorShapeProto;

public class TfSymbolBlock extends AbstractSymbolBlock implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(TfSymbolBlock.class);

    private static final byte VERSION = 1;

    private SavedModelBundle bundle;
    private Session session;
    private SignatureDef servingDefault;
    private PairList<String, Shape> inputDescriptions;
    private PairList<String, Shape> outputDescriptions;
    // store mapping of meaningful key names and actual tensor names used in session
    private ConcurrentHashMap<String, String> inputOutputNames = new ConcurrentHashMap<>();

    public TfSymbolBlock(SavedModelBundle bundle, String signatureDefKey) {
        super(VERSION);
        this.bundle = bundle;
        session = bundle.session();
        MetaGraphDef metaGraphDef = bundle.metaGraphDef();
        Map<String, SignatureDef> signatureDefMap = metaGraphDef.getSignatureDefMap();
        if (signatureDefMap.containsKey(signatureDefKey)) {
            servingDefault = signatureDefMap.get(signatureDefKey);
        } else {
            Set<String> keys = signatureDefMap.keySet();
            logger.warn(
                    "SignatureDefKey: "
                            + signatureDefKey
                            + "not found in Saved Model Bundle."
                            + "Available keys: "
                            + String.join(" ", keys)
                            + "Please use .optOptions(\"SignatureDefKey\", \"value\") with Criteria.builder to load the model."
                            + "Normally the value is \"default\" for TF1.x models and \"serving_default\" for TF2.x models. "
                            + "Refer to: https://www.tensorflow.org/guide/saved_model"
                            + "Loading the model using next available key.");
            servingDefault = signatureDefMap.get(keys.iterator().next());
        }
        describeInput();
        describeOutput();
    }

    /** {@inheritDoc} */
    @Override
    public void removeLastBlock() {
        throw new UnsupportedOperationException("Not supported for TensorFlow Engine");
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        Session.Runner runner = session.runner();
        for (int i = 0; i < inputDescriptions.size(); i++) {
            String inputName = inputDescriptions.get(i).getKey();
            String tensorName = inputOutputNames.get(inputName);

            NDArray inputArray = inputs.get(i);
            // no name specified in input array, use default order from translator
            if (inputArray.getName().isEmpty()) {
                runner.feed(tensorName, ((TfNDArray) inputArray).getTensor());
            } else {
                if (inputArray.getName().equals(inputName)) {
                    runner.feed(tensorName, ((TfNDArray) inputArray).getTensor());
                } else {
                    // find the array with correct name
                    for (NDArray array : inputs) {
                        if (array.getName().equals(inputName)) {
                            runner.feed(tensorName, ((TfNDArray) array).getTensor());
                        }
                    }
                }
            }
        }
        for (int i = 0; i < outputDescriptions.size(); i++) {
            String key = outputDescriptions.get(i).getKey();
            runner.fetch(inputOutputNames.get(key));
        }
        List<Tensor<?>> result = runner.run();
        TfNDManager tfNDManager = (TfNDManager) inputs.head().getManager();
        NDList resultNDList = new NDList();
        for (int i = 0; i < result.size(); i++) {
            try (Tensor<?> tensor = result.get(i)) {
                NDArray array = tfNDManager.create(tensor);
                array.setName(outputDescriptions.get(i).getKey());
                resultNDList.add(array);
            }
        }
        return resultNDList;
    }

    /** {@inheritDoc} */
    @Override
    public void initialize(NDManager manager, DataType dataType, Shape... inputShapes) {
        throw new IllegalStateException("TfSymbolBlock can't be initialized");
    }

    /** {@inheritDoc} */
    @Override
    public boolean isInitialized() {
        return bundle != null;
    }

    /** {@inheritDoc} */
    @Override
    public final PairList<String, Shape> describeInput() {
        if (inputDescriptions == null) {
            inputDescriptions = new PairList<>();
            Map<String, TensorInfo> inputsMap = servingDefault.getInputsMap();
            List<String> keys = new ArrayList<>(inputsMap.keySet());
            Collections.sort(keys);
            for (String key : keys) {
                TensorInfo tensorInfo = inputsMap.get(key);
                TensorShapeProto shapeProto = tensorInfo.getTensorShape();
                inputOutputNames.put(key, tensorInfo.getName());
                inputDescriptions.add(
                        key,
                        new Shape(
                                shapeProto
                                        .getDimList()
                                        .stream()
                                        .mapToLong(TensorShapeProto.Dim::getSize)
                                        .toArray()));
            }
        }
        return inputDescriptions;
    }

    /** {@inheritDoc} */
    @Override
    public final PairList<String, Shape> describeOutput() {
        if (outputDescriptions == null) {
            outputDescriptions = new PairList<>();
            Map<String, TensorInfo> outputsMap = servingDefault.getOutputsMap();
            List<String> keys = new ArrayList<>(outputsMap.keySet());
            Collections.sort(keys);
            for (String key : keys) {
                TensorInfo tensorInfo = outputsMap.get(key);
                TensorShapeProto shapeProto = tensorInfo.getTensorShape();
                // does not support string tensors
                if (tensorInfo.getDtype() == org.tensorflow.proto.framework.DataType.DT_STRING) {
                    continue;
                }
                inputOutputNames.put(key, tensorInfo.getName());
                outputDescriptions.add(
                        key,
                        new Shape(
                                shapeProto
                                        .getDimList()
                                        .stream()
                                        .mapToLong(TensorShapeProto.Dim::getSize)
                                        .toArray()));
            }
        }
        return outputDescriptions;
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes) {
        return new Shape[0];
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (session != null) {
            session.close();
        }
        if (bundle != null) {
            bundle.close();
        }
    }
}
