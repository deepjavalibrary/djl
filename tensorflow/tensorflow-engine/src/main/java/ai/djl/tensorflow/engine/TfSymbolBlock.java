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
import ai.djl.nn.SymbolBlock;
import ai.djl.tensorflow.engine.javacpp.JavacppUtils;
import ai.djl.training.ParameterStore;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.bytedeco.javacpp.Pointer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.internal.c_api.TF_Graph;
import org.tensorflow.internal.c_api.TF_Operation;
import org.tensorflow.internal.c_api.TF_Session;
import org.tensorflow.internal.c_api.TF_Tensor;
import org.tensorflow.proto.framework.MetaGraphDef;
import org.tensorflow.proto.framework.SignatureDef;
import org.tensorflow.proto.framework.TensorInfo;
import org.tensorflow.proto.framework.TensorShapeProto;

/** {@code TfSymbolBlock} is the TensorFlow implementation of {@link SymbolBlock}. */
public class TfSymbolBlock extends AbstractSymbolBlock implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(TfSymbolBlock.class);

    private SavedModelBundle bundle;
    private TF_Graph graphHandle;
    private TF_Session sessionHandle;
    private SignatureDef servingDefault;
    private PairList<String, Shape> inputDescriptions;
    private PairList<String, Shape> outputDescriptions;
    // cached input & output information
    private TF_Operation[] inputOpHandles;
    private int[] inputOpIndices;
    private TF_Operation[] outputOpHandles;
    private int[] outputOpIndices;
    private TF_Operation[] targetOpHandles;

    public TfSymbolBlock(SavedModelBundle bundle, String signatureDefKey) {
        this.bundle = bundle;
        graphHandle = bundle.getGraph();
        sessionHandle = bundle.getSession();
        MetaGraphDef metaGraphDef = bundle.getMetaGraphDef();
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
                            + "Please use .optOption(\"SignatureDefKey\", \"value\") with Criteria.builder to load the model."
                            + "Normally the value is \"default\" for TF1.x models and \"serving_default\" for TF2.x models. "
                            + "Refer to: https://www.tensorflow.org/guide/saved_model"
                            + "Loading the model using next available key.");
            servingDefault = signatureDefMap.get(keys.iterator().next());
        }
        describeInput();
        describeOutput();
        // we don't use target for now
        targetOpHandles = new TF_Operation[0];
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
        TF_Tensor[] inputTensorHandles = new TF_Tensor[inputDescriptions.size()];

        for (int i = 0; i < inputDescriptions.size(); i++) {
            String inputName = inputDescriptions.get(i).getKey();

            TfNDArray currentNDArray = (TfNDArray) inputs.get(i);
            // if no name specified in input array or
            // the input order matches inputDescriptions
            // use default order from translator
            String name = currentNDArray.getName();
            if (name == null || name.isEmpty() || name.equals(inputName)) {
                inputTensorHandles[i] = JavacppUtils.resolveTFETensor(currentNDArray.getHandle());
                continue;
            }
            // for loop to search the right NDArray
            for (NDArray array : inputs) {
                if (array.getName().equals(inputName)) {
                    inputTensorHandles[i] =
                            JavacppUtils.resolveTFETensor(((TfNDArray) array).getHandle());
                }
            }
        }

        TF_Tensor[] outputs =
                JavacppUtils.runSession(
                        sessionHandle,
                        null,
                        inputTensorHandles,
                        inputOpHandles,
                        inputOpIndices,
                        outputOpHandles,
                        outputOpIndices,
                        targetOpHandles);

        TfNDManager tfNDManager = (TfNDManager) inputs.head().getManager();
        NDList resultNDList = new NDList();
        for (int i = 0; i < outputs.length; i++) {
            TfNDArray array = new TfNDArray(tfNDManager, JavacppUtils.createTFETensor(outputs[i]));
            array.setName(outputDescriptions.get(i).getKey());
            resultNDList.add(array);
        }

        // free all unused native resources
        Arrays.stream(inputTensorHandles).forEach(Pointer::close);
        Arrays.stream(outputs).forEach(Pointer::close);
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

            inputOpHandles = new TF_Operation[keys.size()];
            inputOpIndices = new int[keys.size()];
            for (int i = 0; i < keys.size(); ++i) {
                TensorInfo tensorInfo = inputsMap.get(keys.get(i));
                TensorShapeProto shapeProto = tensorInfo.getTensorShape();
                inputDescriptions.add(
                        keys.get(i),
                        new Shape(
                                shapeProto
                                        .getDimList()
                                        .stream()
                                        .mapToLong(TensorShapeProto.Dim::getSize)
                                        .toArray()));
                Pair<TF_Operation, Integer> pair =
                        JavacppUtils.getGraphOperationByName(graphHandle, tensorInfo.getName());
                inputOpHandles[i] = pair.getKey();
                inputOpIndices[i] = pair.getValue();
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

            List<TF_Operation> outputOpHandlesList = new ArrayList<>();
            List<Integer> outputOpIndicesList = new ArrayList<>();
            for (String key : keys) {
                TensorInfo tensorInfo = outputsMap.get(key);
                TensorShapeProto shapeProto = tensorInfo.getTensorShape();
                // does not support string tensors
                if (tensorInfo.getDtype() == org.tensorflow.proto.framework.DataType.DT_STRING) {
                    continue;
                }
                outputDescriptions.add(
                        key,
                        new Shape(
                                shapeProto
                                        .getDimList()
                                        .stream()
                                        .mapToLong(TensorShapeProto.Dim::getSize)
                                        .toArray()));
                Pair<TF_Operation, Integer> pair =
                        JavacppUtils.getGraphOperationByName(graphHandle, tensorInfo.getName());
                outputOpHandlesList.add(pair.getKey());
                outputOpIndicesList.add(pair.getValue());
            }
            outputOpHandles = outputOpHandlesList.toArray(new TF_Operation[0]);
            outputOpIndices = outputOpIndicesList.stream().mapToInt(i -> i).toArray();
        }
        return outputDescriptions;
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[0];
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (bundle != null) {
            bundle.close();
        }
        // free cached input & output native resources
        Arrays.stream(inputOpHandles).forEach(Pointer::close);
        Arrays.stream(outputOpHandles).forEach(Pointer::close);
        Arrays.stream(targetOpHandles).forEach(Pointer::close);
    }
}
