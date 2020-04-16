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
package ai.djl.modality.nlp;

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.BlockList;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * {@code EncoderDecoder} is a general implementation of the very popular encoder-decoder
 * architecture. This class depends on implementations of {@link Encoder} and {@link Decoder} to
 * provide encoder-decoder architecture for different tasks and inputs such as machine
 * translation(text-text), image captioning(image-text) etc.
 */
public class EncoderDecoder extends AbstractBlock {
    protected Encoder encoder;
    protected Decoder decoder;

    /**
     * Constructs a new instance of {@code EncoderDecoder} class with the given {@link Encoder} and
     * {@code Decoder}.
     *
     * @param encoder the {@link Encoder}
     * @param decoder the {@link Decoder}
     */
    public EncoderDecoder(Encoder encoder, Decoder decoder) {
        this.encoder = encoder;
        this.decoder = decoder;
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeInput() {
        if (!isInitialized()) {
            throw new IllegalStateException("Parameter of this block are not initialised");
        }
        inputNames = Arrays.asList("encoderInput", "decoderInput");
        return new PairList<>(inputNames, Arrays.asList(inputShapes));
    }

    /**
     * Applies the forward function of the encoder and the decoder. This method should be called
     * only on blocks that are initialized.
     *
     * @param parameterStore the parameter store
     * @param encoderInputs the input for the encoder
     * @param decoderInputs the input for the decoder
     * @param params optional parameters
     * @return the output of the forward pass
     */
    public NDList forward(
            ParameterStore parameterStore,
            NDList encoderInputs,
            NDList decoderInputs,
            PairList<String, Object> params) {
        NDList encoderOutputs = encoder.forward(parameterStore, encoderInputs, params);
        decoder.initState(encoder.getState(encoderOutputs));
        return decoder.forward(parameterStore, decoderInputs, params);
    }

    /**
     * Applies the forward function of the encoder and the decoder. This method should be called
     * only on blocks that are initialized.
     *
     * <p>This forward function in the {@code EncoderDecoder} class assumes the input {@link NDList}
     * contains both the encoder and decoder inputs. Further, it assumes that the first index in the
     * input {@link NDList} contains the encoder input and the second index contains the decoder
     * input.
     *
     * @param parameterStore the parameter store
     * @param inputs the input NDList
     * @param params optional parameters
     * @return the output of the forward pass
     */
    @Override
    public NDList forward(
            ParameterStore parameterStore, NDList inputs, PairList<String, Object> params) {
        return forward(
                parameterStore, new NDList(inputs.get(0)), new NDList(inputs.get(1)), params);
    }

    /**
     * Applies the forward function of the encoder and the decoder. This method should be called
     * only on blocks that are initialized.
     *
     * <p>This forward function in the {@code EncoderDecoder} class assumes the input {@link NDList}
     * contains both the encoder and decoder inputs. Further, it assumes that the first index in the
     * input {@link NDList} contains the encoder input and the second index contains the decoder
     * input.
     *
     * @param parameterStore the parameter store
     * @param inputs the input NDList
     * @return the output of the forward pass
     */
    @Override
    public NDList forward(ParameterStore parameterStore, NDList inputs) {
        return forward(parameterStore, inputs, null);
    }

    /**
     * Initializes the parameters of the block. This method must be called before calling `forward`.
     *
     * <p>This method assumes that inputShapes contains encoder and decoder inputs in index 0 and 1
     * respectively.
     *
     * @param manager the NDManager to initialize the parameters
     * @param dataType the datatype of the parameters
     * @param inputShapes the shapes of the inputs to the block
     * @return the shapes of the outputs of the block
     */
    @Override
    public Shape[] initialize(NDManager manager, DataType dataType, Shape... inputShapes) {
        beforeInitialize(inputShapes);
        encoder.initialize(manager, dataType, inputShapes[0]);
        return decoder.initialize(manager, dataType, inputShapes[1]);
    }

    /** {@inheritDoc} */
    @Override
    public BlockList getChildren() {
        BlockList children = encoder.getChildren();
        children.addAll(decoder.getChildren());
        return children;
    }

    /** {@inheritDoc} */
    @Override
    public List<Parameter> getDirectParameters() {
        return Collections.emptyList();
    }

    /** {@inheritDoc} */
    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        throw new IllegalArgumentException("EncodeDecoder blocks have no direct parameters");
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes) {
        return decoder.getOutputShapes(manager, new Shape[] {inputShapes[1]});
    }

    /** {@inheritDoc} */
    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        encoder.saveParameters(os);
        decoder.saveParameters(os);
    }

    /** {@inheritDoc} */
    @Override
    public void loadParameters(NDManager manager, DataInputStream is)
            throws IOException, MalformedModelException {
        encoder.loadParameters(manager, is);
        decoder.loadParameters(manager, is);
    }
}
