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

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.translate.Translator;
import ai.djl.util.PairList;
import com.google.protobuf.InvalidProtocolBufferException;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;
import org.tensorflow.framework.TensorInfo;
import org.tensorflow.framework.TensorShapeProto;

public class TfModel implements Model {

    private Path modelDir;
    private SavedModelBundle bundle;
    private AtomicBoolean first = new AtomicBoolean(true);

    private PairList<String, Shape> constructDataDescFromModel(Map<String, TensorInfo> info) {
        PairList<String, Shape> descs = new PairList<>();
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
            descs.add(t.getName(), new Shape(shape));
        }
        return descs;
    }

    public void load(Path modelDir, String... tags) throws InvalidProtocolBufferException {
        if (tags == null || tags.length == 0) {
            tags = new String[] {"serve"};
        }
        bundle = SavedModelBundle.load(modelDir.toString(), tags);
        SignatureDef sig =
                MetaGraphDef.parseFrom(bundle.metaGraphDef())
                        .getSignatureDefOrThrow("serving_default");
        constructDataDescFromModel(sig.getInputsMap());
        constructDataDescFromModel(sig.getOutputsMap());
    }

    /** {@inheritDoc} */
    @Override
    public void load(Path modelPath, String modelName, Map<String, String> options)
            throws IOException, MalformedModelException {
        try {
            load(modelPath);
        } catch (InvalidProtocolBufferException e) {
            throw new IOException(e);
        }
    }

    public void load(String modelDir, byte[] configProto, byte[] runOptions, String... tags) {
        this.modelDir = Paths.get(modelDir);
        bundle =
                SavedModelBundle.loader(modelDir)
                        .withConfigProto(configProto)
                        .withRunOptions(runOptions)
                        .withTags(tags)
                        .load();
    }

    /** {@inheritDoc} */
    @Override
    public void save(Path modelPath, String modelName) {}

    public org.tensorflow.Graph getTensorflowGraph() {
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
    public Block getBlock() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void setBlock(Block block) {}

    /** {@inheritDoc} */
    @Override
    public String getName() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public String getProperty(String key) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void setProperty(String key, String value) {}

    /** {@inheritDoc} */
    @Override
    public Trainer newTrainer(TrainingConfig trainingConfig) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public <I, O> Predictor<I, O> newPredictor(Translator<I, O> translator) {
        return new TfPredictor<>(this, translator, first.getAndSet(false));
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeInput() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeOutput() {
        return null;
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

    /** {@inheritDoc} */
    @Override
    public NDManager getNDManager() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void setDataType(DataType dataType) {}

    /** {@inheritDoc} */
    @Override
    public DataType getDataType() {
        return DataType.FLOAT32;
    }

    /** {@inheritDoc} */
    @Override
    public void cast(DataType dataType) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public void close() {}
}
