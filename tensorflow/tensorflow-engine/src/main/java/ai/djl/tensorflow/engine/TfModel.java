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

import ai.djl.BaseModel;
import ai.djl.Device;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.translate.Translator;
import ai.djl.util.PairList;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;
import org.tensorflow.Graph;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;

public class TfModel extends BaseModel {
    private Path modelDir;
    private SavedModelBundle bundle;
    private AtomicBoolean first = new AtomicBoolean(true);
    private NDManager manager;
    private Session session;
    private Graph graph;

    /**
     * Constructs a new Model on a given device.
     *
     * @param device the device the model should be located on
     */
    TfModel(Device device) {
        device = Device.defaultIfNull(device);
        properties = new ConcurrentHashMap<>();
        manager = TfNDManager.getSystemManager().newSubManager(device);
        first = new AtomicBoolean(true);
    }

    public void load(Path modelDir, String... tags) {
        if (tags == null || tags.length == 0) {
            tags = new String[] {"serve"};
        }
        bundle = SavedModelBundle.load(modelDir.toString(), tags);
        graph = bundle.graph();
        session = bundle.session();
    }

    /** {@inheritDoc} */
    @Override
    public void load(Path modelPath, String modelName, Map<String, String> options) {
        String[] tags;
        if (options == null || options.isEmpty()) {
            tags = new String[] {"serve"};
        } else {
            tags = options.values().toArray(new String[] {});
        }
        load(modelPath, tags);
    }

    public void load(String modelDir, byte[] configProto, byte[] runOptions, String... tags) {
        this.modelDir = Paths.get(modelDir);
        bundle =
                SavedModelBundle.loader(modelDir)
                        .withConfigProto(configProto)
                        .withRunOptions(runOptions)
                        .withTags(tags)
                        .load();
        graph = bundle.graph();
        session = bundle.session();
    }

    /** {@inheritDoc} */
    @Override
    public void save(Path modelPath, String modelName) {}

    public Graph getGraph() {
        return graph;
    }

    public Session getSession() {
        return session;
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
        return manager;
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
    public void close() {
        manager.close();
        if (bundle != null) {
            bundle.close();
        }
        if (graph != null) {
            graph.close();
        }
        if (session != null) {
            session.close();
        }
        bundle = null;
        graph = null;
        session = null;
    }
}
