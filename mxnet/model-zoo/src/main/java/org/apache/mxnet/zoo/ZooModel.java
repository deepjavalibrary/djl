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
package org.apache.mxnet.zoo;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.function.Function;
import software.amazon.ai.Context;
import software.amazon.ai.Model;
import software.amazon.ai.inference.Predictor;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.nn.Block;
import software.amazon.ai.translate.Translator;

public class ZooModel<I, O> implements Model {

    private Model model;
    private Translator<I, O> translator;

    public ZooModel(Model model, Translator<I, O> translator) {
        this.model = model;
        this.translator = translator;
    }

    public Predictor<I, O> newPredictor() {
        return newPredictor(translator, null);
    }

    public Predictor<I, O> newPredictor(Context context) {
        return newPredictor(translator, context);
    }

    /** {@inheritDoc} */
    @Override
    public <P, Q> Predictor<P, Q> newPredictor(Translator<P, Q> translator, Context context) {
        return model.newPredictor(translator, context);
    }

    public Translator<I, O> getTranslator() {
        return translator;
    }

    public Model quantize() {
        return model.cast(DataType.UINT8);
    }

    /** {@inheritDoc} */
    @Override
    public DataDesc[] describeInput() {
        return model.describeInput();
    }

    /** {@inheritDoc} */
    @Override
    public DataDesc[] describeOutput() {
        return model.describeOutput();
    }

    /** {@inheritDoc} */
    @Override
    public String[] getArtifactNames() {
        return model.getArtifactNames();
    }

    /** {@inheritDoc} */
    @Override
    public <T> T getArtifact(String name, Function<InputStream, T> function) throws IOException {
        return model.getArtifact(name, function);
    }

    /** {@inheritDoc} */
    @Override
    public URL getArtifact(String name) throws IOException {
        return model.getArtifact(name);
    }

    /** {@inheritDoc} */
    @Override
    public InputStream getArtifactAsStream(String name) throws IOException {
        return model.getArtifactAsStream(name);
    }

    /** {@inheritDoc} */
    @Override
    public Block getBlock() {
        return model.getBlock();
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getManager() {
        return model.getManager();
    }

    /** {@inheritDoc} */
    @Override
    public Model cast(DataType dataType) {
        return model.cast(dataType);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        model.close();
    }
}
