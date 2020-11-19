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
import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Block;
import com.google.protobuf.InvalidProtocolBufferException;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.proto.framework.ConfigProto;
import org.tensorflow.proto.framework.RunOptions;

public class TfModel extends BaseModel {

    /**
     * Constructs a new Model on a given device.
     *
     * @param name the model name
     * @param device the device the model should be located on
     */
    TfModel(String name, Device device) {
        super(name);
        device = Device.defaultIfNull(device);
        properties = new ConcurrentHashMap<>();
        manager = TfNDManager.getSystemManager().newSubManager(device);
        manager.setName("tfModel");
    }

    /** {@inheritDoc} */
    @Override
    public void load(Path modelPath, String prefix, Map<String, ?> options)
            throws FileNotFoundException, MalformedModelException {
        modelDir = modelPath.toAbsolutePath();
        if (prefix == null) {
            prefix = modelName;
        }
        Path exportDir = findModelDir(prefix);
        if (exportDir == null) {
            exportDir = findModelDir("saved_model.pb");
            if (exportDir == null) {
                throw new FileNotFoundException("No TensorFlow model found in: " + modelDir);
            }
        }
        String[] tags = null;
        ConfigProto configProto = null;
        RunOptions runOptions = null;
        String signatureDefKey = "serving_default";
        if (options != null) {
            Object tagOption = options.get("Tags");
            if (tagOption instanceof String[]) {
                tags = (String[]) tagOption;
            } else if (tagOption instanceof String) {
                if (((String) tagOption).isEmpty()) {
                    tags = new String[0];
                } else {
                    tags = ((String) tagOption).split(",");
                }
            }
            Object config = options.get("ConfigProto");
            if (config instanceof ConfigProto) {
                configProto = (ConfigProto) config;
            } else if (config instanceof String) {
                try {
                    byte[] buf = Base64.getDecoder().decode((String) config);
                    configProto = ConfigProto.parseFrom(buf);
                } catch (InvalidProtocolBufferException e) {
                    throw new MalformedModelException("Invalid ConfigProto: " + config, e);
                }
            }
            Object run = options.get("RunOptions");
            if (run instanceof RunOptions) {
                runOptions = (RunOptions) run;
            } else if (run instanceof String) {
                try {
                    byte[] buf = Base64.getDecoder().decode((String) run);
                    runOptions = RunOptions.parseFrom(buf);
                } catch (InvalidProtocolBufferException e) {
                    throw new MalformedModelException("Invalid RunOptions: " + run, e);
                }
            }
            signatureDefKey = (String) options.get("SignatureDefKey");
        }
        if (tags == null) {
            tags = new String[] {"serve"};
        }

        SavedModelBundle.Loader loader = SavedModelBundle.loader(exportDir.toString());
        if (tags.length > 0) {
            loader.withTags(tags);
        } else {
            // FIXME: workaround TF-java bug
            try {
                Field field = SavedModelBundle.Loader.class.getDeclaredField("tags");
                field.setAccessible(true);
                field.set(loader, tags);
            } catch (ReflectiveOperationException e) {
                throw new AssertionError(e);
            }
        }
        if (configProto != null) {
            loader.withConfigProto(configProto);
        }
        if (runOptions != null) {
            loader.withRunOptions(runOptions);
        }

        SavedModelBundle bundle = loader.load();
        block = new TfSymbolBlock(bundle, signatureDefKey);
    }

    private Path findModelDir(String prefix) {
        Path path = modelDir.resolve(prefix);
        if (!Files.exists(path)) {
            return null;
        }
        if (Files.isRegularFile(path)) {
            return modelDir;
        } else if (Files.isDirectory(path)) {
            Path file = path.resolve("saved_model.pb");
            if (Files.exists(file) && Files.isRegularFile(file)) {
                return path;
            }
        }
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void save(Path modelPath, String newModelName) {
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
    public void close() {
        super.close();
        if (block != null) {
            block.clear();
        }
    }
}
