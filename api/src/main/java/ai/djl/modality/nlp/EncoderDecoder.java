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
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;

/**
 * {@code EncoderDecoder} is a general implementation of the very popular encoder-decoder
 * architecture. This class depends on implementations of {@link Encoder} and {@link Decoder} to
 * provide encoder-decoder architecture for different tasks and inputs such as machine
 * translation(text-text), image captioning(image-text) etc.
 */
public class EncoderDecoder extends AbstractBlock {

    private static final byte VERSION = 1;

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
        super(VERSION);
        this.encoder = addChildBlock("Encoder", encoder);
        this.decoder = addChildBlock("Decoder", decoder);
        inputNames = Arrays.asList("encoderInput", "decoderInput");
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeInput() {
        if (!isInitialized()) {
            throw new IllegalStateException("Parameter of this block are not initialised");
        }
        return new PairList<>(inputNames, Arrays.asList(inputShapes));
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        if (training) {
            throw new IllegalArgumentException("You must use forward with labels when training");
        }
        throw new UnsupportedOperationException(
                "EncoderDecoder prediction has not been implemented yet");
    }

    /** {@inheritDoc} */
    @Override
    public NDList forward(
            ParameterStore parameterStore,
            NDList data,
            NDList labels,
            PairList<String, Object> params) {
        NDList encoderOutputs = encoder.forward(parameterStore, data, true, params);
        // add hidden states & cell states to decoder inputs
        labels.addAll(encoder.getStates(encoderOutputs));
        return decoder.forward(parameterStore, labels, true, params);
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
     */
    @Override
    public void initialize(NDManager manager, DataType dataType, Shape... inputShapes) {
        beforeInitialize(inputShapes);
        encoder.initialize(manager, dataType, inputShapes[0]);
        decoder.initialize(manager, dataType, inputShapes[1]);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return decoder.getOutputShapes(new Shape[] {inputShapes[1]});
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
