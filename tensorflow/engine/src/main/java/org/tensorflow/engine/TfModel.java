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
package org.tensorflow.engine;

import com.google.protobuf.InvalidProtocolBufferException;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.function.Function;
import org.tensorflow.Graph;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;
import org.tensorflow.framework.TensorInfo;
import org.tensorflow.framework.TensorShapeProto;
import software.amazon.ai.Context;
import software.amazon.ai.Model;
import software.amazon.ai.inference.Predictor;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.Block;
import software.amazon.ai.training.Trainer;
import software.amazon.ai.training.initializer.Initializer;
import software.amazon.ai.training.optimizer.Optimizer;
import software.amazon.ai.translate.TrainTranslator;
import software.amazon.ai.translate.Translator;

public class TfModel implements Model {

    private Path modelDir;
    private SavedModelBundle bundle;
    private DataDesc[] inputDesc;
    private DataDesc[] outputDesc;

    TfModel(Path modelDir, SavedModelBundle bundle) throws InvalidProtocolBufferException {
        this.modelDir = modelDir;
        this.bundle = bundle;
        SignatureDef sig =
                MetaGraphDef.parseFrom(this.bundle.metaGraphDef())
                        .getSignatureDefOrThrow("serving_default");
        inputDesc = constructDataDescFromModel(sig.getInputsMap());
        outputDesc = constructDataDescFromModel(sig.getOutputsMap());
    }

    private DataDesc[] constructDataDescFromModel(Map<String, TensorInfo> info) {
        DataDesc[] descs = new DataDesc[info.size()];
        int dataDescIter = 0;
        for (Map.Entry<String, TensorInfo> entry : info.entrySet()) {
            TensorInfo t = entry.getValue();
            // StringBuilder layout = new StringBuilder();
            long[] shape = new long[t.getTensorShape().getDimCount()];
            int dimIter = 0;
            for (TensorShapeProto.Dim dim : t.getTensorShape().getDimList()) {
                // layout.append(dim.getName());
                shape[dimIter] = dim.getSize();
                dimIter++;
            }
            // TODO: Add DataType mapping from framework.DataType
            // TODO: Add Layout mapping for the layout
            descs[dataDescIter] = new DataDesc(new Shape(shape), null, t.getName());
            dataDescIter++;
        }
        return descs;
    }

    public static TfModel load(String modelDir, String... tags)
            throws InvalidProtocolBufferException {
        if (tags == null || tags.length == 0) {
            tags = new String[] {"serve"};
        }
        return new TfModel(Paths.get(modelDir), SavedModelBundle.load(modelDir, tags));
    }

    public static TfModel load(
            String modelDir, byte[] configProto, byte[] runOptions, String... tags)
            throws InvalidProtocolBufferException {
        SavedModelBundle bundle =
                SavedModelBundle.loader(modelDir)
                        .withConfigProto(configProto)
                        .withRunOptions(runOptions)
                        .withTags(tags)
                        .load();
        return new TfModel(Paths.get(modelDir), bundle);
    }

    public Graph getGraph() {
        return bundle.graph();
    }

    public Session getSession() {
        return bundle.session();
    }

    private byte[] getMetaGraphDef() {
        return bundle.metaGraphDef();
    }

    /** {@inheritDoc} */
    @Override
    public <I, L, O> Trainer<I, L, O> newTrainer(TrainTranslator<I, L, O> trainTranslator) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public <I, L, O> Trainer<I, L, O> newTrainer(
            TrainTranslator<I, L, O> trainTranslator, Optimizer optimizer) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public <I, L, O> Trainer<I, L, O> newTrainer(
            TrainTranslator<I, L, O> trainTranslator, Optimizer optimizer, Context context) {
        return null;
    }

    @Override
    public void setInitializer(Initializer initializer) {}

    @Override
    public void setInitializer(Initializer initializer, boolean overwrite) {}

    /** {@inheritDoc} */
    @Override
    public <I, O> Predictor<I, O> newPredictor(Translator<I, O> translator, Context context) {
        return new TfPredictor<>(this, translator);
    }

    /** {@inheritDoc} */
    @Override
    public DataDesc[] describeInput() {
        return inputDesc;
    }

    /** {@inheritDoc} */
    @Override
    public DataDesc[] describeOutput() {
        return outputDesc;
    }

    /** {@inheritDoc} */
    @Override
    public String[] getArtifactNames() {
        return new String[0];
    }

    /** {@inheritDoc} */
    @Override
    public <T> T getArtifact(String name, Function<InputStream, T> function) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public URL getArtifact(String artifactName) throws IOException {
        if (artifactName == null) {
            throw new IllegalArgumentException("artifactName cannot be null");
        }
        Path file = modelDir.resolve(artifactName);
        if (Files.exists(file) && Files.isReadable(file)) {
            return file.toUri().toURL();
        }
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public InputStream getArtifactAsStream(String name) {
        return null;
    }

    @Override
    public Block getBlock() {
        return null;
    }

    @Override
    public NDManager getManager() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Model cast(DataType dataType) {
        return null;
    }

    @Override
    public void close() {}
}
