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
package org.apache.mxnet.engine;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.apache.mxnet.jna.JnaUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.ai.Model;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.util.Pair;
import software.amazon.ai.util.PairList;
import software.amazon.ai.util.Utils;

/**
 * {@code MxModel} is MXNet implementation of {@link Model}.
 *
 * <p>MxModel contains all methods in Model to load and process model. In addition, it provides
 * MXNet Specific functionality, such as getSymbol to obtain the Symbolic graph and getParameters to
 * obtain the parameter NDArrays
 */
public class MxModel implements Model {

    private static final Logger logger = LoggerFactory.getLogger(MxModel.class);

    private NDManager manager;
    private Path modelDir;
    private Symbol symbol;
    private PairList<String, MxNDArray> parameters;
    private String[] optimizerStates;
    private DataDesc[] inputData;
    private Map<String, Object> artifacts = new ConcurrentHashMap<>();

    MxModel(
            NDManager manager,
            Path modelDir,
            Symbol symbol,
            PairList<String, MxNDArray> parameters,
            String[] optimizerStates) {
        this.manager = manager;
        this.modelDir = modelDir;
        this.symbol = symbol;
        this.parameters = parameters;
        this.optimizerStates = optimizerStates;
    }

    static MxModel loadModel(String prefix, int epoch) throws IOException {
        return loadModel(MxNDManager.SYSTEM_MANAGER.newSubManager(), prefix, epoch);
    }

    static MxModel loadModel(MxNDManager manager, String prefix, int epoch) throws IOException {
        // TODO: Find a better solution to get rid of this line
        JnaUtils.getAllOpNames();
        Symbol symbol = Symbol.load(manager, prefix + "-symbol.json");
        String paramFile = String.format("%s-%04d.params", prefix, epoch);
        String stateFile = String.format("%s-%04d.states", prefix, epoch);
        Path modelDir = Paths.get(paramFile).toAbsolutePath().getParent();

        PointerByReference namesRef = new PointerByReference();
        Pointer[] handles = JnaUtils.loadNdArray(paramFile, namesRef);
        String[] names = namesRef.getValue().getStringArray(0, handles.length);

        PairList<String, MxNDArray> parameters = new PairList<>();

        for (int i = 0; i < names.length; ++i) {
            String[] pair = names[i].split(":", 2);
            MxNDArray array = manager.create(handles[i]);
            parameters.add(pair[1], array);
        }

        String[] stateNames = Utils.readLines(Paths.get(stateFile)).toArray(JnaUtils.EMPTY_ARRAY);
        // TODO: Check if Symbol has all names that params file have
        return new MxModel(manager, modelDir, symbol, parameters, stateNames);
    }

    /**
     * Returns the Symbolic graph from the model.
     *
     * @return {@link Symbol} object
     */
    public Symbol getSymbol() {
        return symbol;
    }

    /**
     * Returns the parameter Pairs from the model.
     *
     * <p>The parameter follow the format as: "name : paramWeight"
     *
     * @return a {@link PairList} of model weights with their name
     */
    public PairList<String, MxNDArray> getParameters() {
        return new PairList<>(parameters.keys(), parameters.values());
    }

    /** {@inheritDoc} */
    @Override
    public Model cast(DataType dataType) {
        if (parameters.get(0).getValue().getDataType() == dataType) {
            logger.debug("You are casting the model to its original type!");
            return this;
        }

        // TODO: This implementation is unsafe, new Model shares the same
        // symbol and optimizerStates with original one. Close either one
        // will cause anther model instance invalidated.
        PairList<String, MxNDArray> newParam = new PairList<>();
        for (Pair<String, MxNDArray> pair : parameters) {
            newParam.add(pair.getKey(), pair.getValue().asType(dataType, true));
        }
        NDManager newManager = MxNDManager.getSystemManager().newSubManager();
        return new MxModel(newManager, modelDir, symbol, newParam, optimizerStates);
    }

    /** {@inheritDoc} */
    @Override
    public DataDesc[] describeInput() {
        if (inputData == null) {
            String[] allNames = symbol.getAllNames();
            Map<String, Integer> map = new ConcurrentHashMap<>(allNames.length * 3 / 2);
            int index = 0;
            for (String name : allNames) {
                map.put(name, index++);
            }
            for (String name : parameters.keys()) {
                map.remove(name);
            }
            inputData = new DataDesc[map.size()];

            index = 0;
            for (String name : map.keySet()) {
                inputData[index++] = new DataDesc(new Shape(), name);
            }
        }
        return inputData;
    }

    /** {@inheritDoc} */
    @Override
    public DataDesc[] describeOutput() {
        return null;
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
    @SuppressWarnings("unchecked")
    @Override
    public <T> T getArtifact(String name, Function<InputStream, T> function) throws IOException {
        try {
            Object artifact =
                    artifacts.computeIfAbsent(
                            name,
                            v -> {
                                try (InputStream is = getArtifactAsStream(name)) {
                                    return function.apply(is);
                                } catch (IOException e) {
                                    throw new IllegalStateException(e);
                                }
                            });
            return (T) artifact;
        } catch (RuntimeException e) {
            Throwable t = e.getCause();
            if (t instanceof IOException) {
                throw (IOException) e.getCause();
            }
            throw e;
        }
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
    public InputStream getArtifactAsStream(String name) throws IOException {
        URL url = getArtifact(name);
        if (url == null) {
            return null;
        }
        return url.openStream();
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        manager.close();
    }

    /** {@inheritDoc} */
    @SuppressWarnings("deprecation")
    @Override
    protected void finalize() throws Throwable {
        if (((MxNDManager) manager).isOpen()) {
            if (logger.isDebugEnabled()) {
                logger.warn("Model was not closed explicitly: {}", getClass().getSimpleName());
            }
            close();
        }
        super.finalize();
    }
}
