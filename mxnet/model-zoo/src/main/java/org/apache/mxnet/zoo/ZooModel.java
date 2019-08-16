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
import software.amazon.ai.Block;
import software.amazon.ai.Context;
import software.amazon.ai.Model;
import software.amazon.ai.inference.Predictor;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.translate.Translator;

public class ZooModel<I, O> implements Model {

    private Model model;
    private Translator<I, O> translator;

    public ZooModel(Model model, Translator<I, O> translator) {
        this.model = model;
        this.translator = translator;
    }

    public Predictor<I, O> newPredictor() {
        return newPredictor(Context.defaultContext());
    }

    public Predictor<I, O> newPredictor(Context context) {
        return Predictor.newInstance(model, translator, context);
    }

    public Translator<I, O> getTranslator() {
        return translator;
    }

    public Model quantize() {
        return model.cast(DataType.UINT8);
    }

    @Override
    public DataDesc[] describeInput() {
        return model.describeInput();
    }

    @Override
    public DataDesc[] describeOutput() {
        return model.describeOutput();
    }

    @Override
    public String[] getArtifactNames() {
        return model.getArtifactNames();
    }

    @Override
    public <T> T getArtifact(String name, Function<InputStream, T> function) throws IOException {
        return model.getArtifact(name, function);
    }

    @Override
    public URL getArtifact(String name) throws IOException {
        return model.getArtifact(name);
    }

    @Override
    public InputStream getArtifactAsStream(String name) throws IOException {
        return model.getArtifactAsStream(name);
    }

    @Override
    public Block getBlock() {
        return model.getBlock();
    }

    @Override
    public Model cast(DataType dataType) {
        return model.cast(dataType);
    }

    @Override
    public void close() {
        model.close();
    }
}
