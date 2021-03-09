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
package ai.djl.mxnet.engine;

import ai.djl.BaseModel;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.mxnet.jna.JnaUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.nn.Parameter;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.initializer.Initializer;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@code MxModel} is the MXNet implementation of {@link Model}.
 *
 * <p>MxModel contains all the methods in Model to load and process a model. In addition, it
 * provides MXNet Specific functionality, such as getSymbol to obtain the Symbolic graph and
 * getParameters to obtain the parameter NDArrays
 */
public class MxModel extends BaseModel {

    private static final Logger logger = LoggerFactory.getLogger(MxModel.class);

    /**
     * Constructs a new Model on a given device.
     *
     * @param name the model name
     * @param device the device the model should be located on
     */
    MxModel(String name, Device device) {
        super(name);
        dataType = DataType.FLOAT32;
        properties = new ConcurrentHashMap<>();
        manager = MxNDManager.getSystemManager().newSubManager(device);
        manager.setName("mxModel");
    }

    /**
     * Loads the MXNet model from a specified location.
     *
     * <p>MXNet engine looks for {MODEL_NAME}-symbol.json and {MODEL_NAME}-{EPOCH}.params files in
     * the specified directory. By default, MXNet engine will pick up the latest epoch of the
     * parameter file. However, users can explicitly specify an epoch to be loaded:
     *
     * <pre>
     * Map&lt;String, String&gt; options = new HashMap&lt;&gt;()
     * <b>options.put("epoch", "3");</b>
     * model.load(modelPath, "squeezenet", options);
     * </pre>
     *
     * @param modelPath the directory of the model
     * @param prefix the model file name or path prefix
     * @param options load model options, see documentation for the specific engine
     * @throws IOException Exception for file loading
     */
    @Override
    public void load(Path modelPath, String prefix, Map<String, ?> options)
            throws IOException, MalformedModelException {
        modelDir = modelPath.toAbsolutePath();
        if (prefix == null) {
            prefix = modelName;
        }
        Path paramFile = paramPathResolver(prefix, options);
        if (paramFile == null) {
            prefix = modelDir.toFile().getName();
            paramFile = paramPathResolver(prefix, options);
            if (paramFile == null) {
                throw new FileNotFoundException(
                        "Parameter file with prefix: " + prefix + " not found in: " + modelDir);
            }
        }

        if (block == null) {
            // load MxSymbolBlock
            Path symbolFile = modelDir.resolve(prefix + "-symbol.json");
            if (Files.notExists(symbolFile)) {
                throw new FileNotFoundException(
                        "Symbol file not found: "
                                + symbolFile
                                + ", please set block manually for imperative model.");
            }
            Symbol symbol =
                    Symbol.load((MxNDManager) manager, symbolFile.toAbsolutePath().toString());
            // TODO: change default name "data" to model-specific one
            block = new MxSymbolBlock(manager, symbol);
        }
        loadParameters(paramFile, options);
        // TODO: Check if Symbol has all names that params file have
        if (options != null && options.containsKey("MxOptimizeFor")) {
            String optimization = (String) options.get("MxOptimizeFor");
            ((MxSymbolBlock) block).optimizeFor(optimization);
        }
    }

    /** {@inheritDoc} */
    @Override
    public Trainer newTrainer(TrainingConfig trainingConfig) {
        PairList<Initializer, Predicate<Parameter>> initializer = trainingConfig.getInitializers();
        if (block == null) {
            throw new IllegalStateException(
                    "You must set a block for the model before creating a new trainer");
        }
        for (Pair<Initializer, Predicate<Parameter>> pair : initializer) {
            if (pair.getKey() != null && pair.getValue() != null) {
                block.setInitializer(pair.getKey(), pair.getValue());
            }
        }

        return new Trainer(this, trainingConfig);
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
                if (fileName.endsWith(".params") || fileName.endsWith("-symbol.json")) {
                    // ignore symbol and param files.
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
        // TODO workaround for MXNet Engine crash issue
        JnaUtils.waitAll();
        super.close();
    }

    @SuppressWarnings("PMD.UseConcurrentHashMap")
    private void loadParameters(Path paramFile, Map<String, ?> options)
            throws IOException, MalformedModelException {
        if (readParameters(paramFile, options)) {
            return;
        }
        logger.debug("DJL formatted model not found, try to find MXNet model");
        NDList paramNDlist = manager.load(paramFile);

        MxSymbolBlock symbolBlock = (MxSymbolBlock) block;

        List<Parameter> parameters = symbolBlock.getAllParameters();
        Map<String, Parameter> map = new LinkedHashMap<>();
        parameters.forEach(p -> map.put(p.getName(), p));

        for (NDArray nd : paramNDlist) {
            String key = nd.getName();
            if (key == null) {
                throw new IllegalArgumentException("Array names must be present in parameter file");
            }

            String paramName = key.split(":", 2)[1];
            Parameter parameter = map.remove(paramName);
            parameter.setArray(nd);
        }
        symbolBlock.setInputNames(new ArrayList<>(map.keySet()));

        // TODO: Find a better to infer model DataType from SymbolBlock.
        dataType = paramNDlist.head().getDataType();
        logger.debug("MXNet Model {} ({}) loaded successfully.", paramFile, dataType);
    }
}
