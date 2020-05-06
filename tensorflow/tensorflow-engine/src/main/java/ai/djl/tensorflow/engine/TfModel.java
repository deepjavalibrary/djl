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
import ai.djl.nn.Block;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.translate.Translator;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.proto.framework.ConfigProto;
import org.tensorflow.proto.framework.RunOptions;

public class TfModel extends BaseModel {

    private AtomicBoolean first;
    private NDManager manager;

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

    /** {@inheritDoc} */
    @Override
    public void load(Path modelPath, String modelName, Map<String, Object> options) {
        modelDir = modelPath.toAbsolutePath();
        String[] tags = null;
        ConfigProto proto = null;
        RunOptions runOptions = null;
        if (options != null) {
            tags = (String[]) options.get("Tags");
            proto = (ConfigProto) options.get("ConfigProto");
            runOptions = (RunOptions) options.get("RunOptions");
        }
        if (tags == null) {
            tags = new String[] {"serve"};
        }

        SavedModelBundle.Loader loader =
                SavedModelBundle.loader(modelDir.toString()).withTags(tags);
        if (proto != null) {
            loader.withConfigProto(proto);
        }
        if (runOptions != null) {
            loader.withRunOptions(runOptions);
        }

        SavedModelBundle bundle = loader.load();
        block = new TfSymbolBlock(manager, bundle);
    }

    /** {@inheritDoc} */
    @Override
    public void save(Path modelPath, String modelName) {
        throw new UnsupportedOperationException("Not supported for TensorFlow Engine");
    }

    /** {@inheritDoc} */
    @Override
    public Block getBlock() {
        return block;
    }

    /** {@inheritDoc} */
    @Override
    public void setBlock(Block block) {
        throw new UnsupportedOperationException("Not supported for TensorFlow Engine");
    }

    /** {@inheritDoc} */
    @Override
    public Trainer newTrainer(TrainingConfig trainingConfig) {
        throw new UnsupportedOperationException("Not supported for TensorFlow Engine");
    }

    /** {@inheritDoc} */
    @Override
    public <I, O> Predictor<I, O> newPredictor(Translator<I, O> translator) {
        return new Predictor<>(this, translator, first.getAndSet(false));
    }
    /** {@inheritDoc} */
    @Override
    public NDManager getNDManager() {
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public String[] getArtifactNames() {
        try {
            List<Path> files =
                    Files.walk(modelDir).filter(Files::isRegularFile).collect(Collectors.toList());
            List<String> ret = new ArrayList<>(files.size());
            for (Path path : files) {
                String fileName = path.toFile().getName();
                if (fileName.endsWith(".pb")) {
                    // ignore model files.
                    continue;
                }
                Path relative = modelDir.relativize(path);
                ret.add(relative.toString());
            }
            return ret.toArray(new String[0]);
        } catch (IOException e) {
            throw new AssertionError("Failed list files", e);
        }
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
        if (block != null) {
            block.clear();
        }
    }
}
