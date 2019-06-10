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

import com.amazon.ai.Model;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.DataType;
import com.amazon.ai.ndarray.types.Shape;
import com.amazon.ai.util.Pair;
import com.amazon.ai.util.PairList;
import com.amazon.ai.util.Utils;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import org.apache.commons.io.FileUtils;
import org.apache.mxnet.jna.JnaUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <code>MxModel</code> is MXNet implementation of {@link Model}.
 *
 * <p>MxModel contains all methods in Model to load and process model. In addition, it provides
 * MXNet Specific functionality, such as getSymbol to obtain the Symbolic graph and getParameters to
 * obtain the parameter NDArrays
 */
public class MxModel implements Model, AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(MxModel.class);

    private File modelDir;
    private Symbol symbol;
    private PairList<String, MxNDArray> parameters;
    private String[] optimizerStates;
    private String[] fixedParameters;
    private DataDesc[] inputData;
    private Map<String, Object> artifacts = new ConcurrentHashMap<>();

    MxModel(
            File modelDir,
            Symbol symbol,
            PairList<String, MxNDArray> parameters,
            String[] optimizerStates) {
        this.modelDir = modelDir;
        this.symbol = symbol;
        this.parameters = parameters;
        this.optimizerStates = optimizerStates;
    }

    static MxModel loadModel(String prefix, int epoch) throws IOException {
        return loadModel(MxNDFactory.SYSTEM_FACTORY, prefix, epoch);
    }

    static MxModel loadModel(MxNDFactory factory, String prefix, int epoch) throws IOException {
        // TODO: Find a better solution to get rid of this line
        JnaUtils.getAllOpNames();
        Symbol symbol = Symbol.load(factory, prefix + "-symbol.json");
        String paramFile = String.format("%s-%04d.params", prefix, epoch);
        String stateFile = String.format("%s-%04d.states", prefix, epoch);
        File modelDir = new File(paramFile).getParentFile();

        PointerByReference namesRef = new PointerByReference();
        Pointer[] handles = JnaUtils.loadNdArray(paramFile, namesRef);
        String[] names = namesRef.getValue().getStringArray(0, handles.length);

        PairList<String, MxNDArray> parameters = new PairList<>();

        for (int i = 0; i < names.length; ++i) {
            String[] pair = names[i].split(":", 2);
            MxNDArray array = factory.create(handles[i]);
            parameters.add(pair[1], array);
        }

        String[] stateNames = JnaUtils.readLines(new File(stateFile)).toArray(JnaUtils.EMPTY_ARRAY);

        return new MxModel(modelDir, symbol, parameters, stateNames);
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
        return parameters;
    }

    /** {@inheritDoc} */
    @Override
    public Model cast(DataType dataType) {
        if (parameters.get(0).getValue().getDataType() == dataType) {
            logger.info("You are casting the model to its original type!");
            return this;
        }
        PairList<String, MxNDArray> newParam = new PairList<>();
        for (Pair<String, MxNDArray> pair : parameters) {
            newParam.add(pair.getKey(), pair.getValue().asType(dataType, true));
        }
        return new MxModel(modelDir, symbol, newParam, optimizerStates);
    }

    public String[] getOptimizerStates() {
        return optimizerStates;
    }

    public void setOptimizerStates(String[] optimizerStates) {
        validate(optimizerStates, "state", true);
        this.optimizerStates = optimizerStates;
    }

    public String[] getFixedParameters() {
        return fixedParameters;
    }

    public void setFixedParameters(String[] fixedParameters) {
        validate(fixedParameters, "fixed_param", true);
        this.fixedParameters = fixedParameters;
    }

    public void saveCheckpoint(
            String prefix,
            int epoch,
            Symbol symbol,
            Map<String, MxNDArray> argParams,
            Map<String, MxNDArray> auxParams) {
        symbol.save(prefix + "-symbol.json");
        String paramName = String.format("%s-%04d.params", prefix, epoch);

        Pointer[] pointers = new Pointer[argParams.size() + auxParams.size()];
        String[] keys = new String[pointers.length];
        int i = 0;
        for (Map.Entry<String, MxNDArray> entry : argParams.entrySet()) {
            keys[i] = "arg:" + entry.getKey();
            pointers[i] = entry.getValue().getHandle();
            ++i;
        }
        for (Map.Entry<String, MxNDArray> entry : auxParams.entrySet()) {
            keys[i] = "aux:" + entry.getKey();
            pointers[i] = entry.getValue().getHandle();
            ++i;
        }

        JnaUtils.saveNdArray(paramName, pointers, keys);
    }

    public void save(File dir, String name, int epoch) {}

    /** {@inheritDoc} */
    @Override
    public void close() {
        symbol.close();

        for (int i = 0; i < parameters.size(); ++i) {
            MxNDArray array = parameters.valueAt(i);
            array.close();
        }
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
        Collection<File> files = FileUtils.listFiles(modelDir, null, true);
        List<String> ret = new ArrayList<>(files.size());
        Path base = modelDir.toPath();
        for (File f : files) {
            String fileName = f.getName();
            if (fileName.endsWith(".params") || fileName.endsWith("-symbol.json")) {
                // ignore symbol and param files.
                continue;
            }
            Path path = f.toPath();
            Path relative = base.relativize(path);
            ret.add(relative.toString());
        }
        return ret.toArray(new String[0]);
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
        File file = new File(modelDir, artifactName);
        if (file.exists() && file.canRead()) {
            return file.toURI().toURL();
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

    private void validate(String[] names, String typeName, boolean required) {
        if (names == null || names.length == 0) {
            return;
        }

        String[] args = symbol.getArgParams();
        for (String name : names) {
            if (!Utils.contains(args, name)) {
                String msg =
                        String.format(
                                "Input %s_%s is not found in symbol.list_arguments().",
                                typeName, name);
                if (required) {
                    throw new IllegalArgumentException(msg);
                }
                logger.warn(msg);
            }
        }
    }
}
