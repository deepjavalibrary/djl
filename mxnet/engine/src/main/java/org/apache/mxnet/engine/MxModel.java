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
import org.apache.mxnet.nn.MxSymbolBlock;
import software.amazon.ai.Context;
import software.amazon.ai.Model;
import software.amazon.ai.inference.Predictor;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.nn.Block;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.translate.Translator;
import software.amazon.ai.util.Pair;

/**
 * {@code MxModel} is MXNet implementation of {@link Model}.
 *
 * <p>MxModel contains all methods in Model to load and process model. In addition, it provides
 * MXNet Specific functionality, such as getSymbol to obtain the Symbolic graph and getParameters to
 * obtain the parameter NDArrays
 */
public class MxModel implements Model {

    private Path modelDir;
    private Block block;
    private DataDesc[] inputData;
    private MxNDManager manager;
    private Map<String, Object> artifacts = new ConcurrentHashMap<>();

    MxModel(Path modelDir, Block block, MxNDManager manager) {
        this.modelDir = modelDir;
        this.block = block;
        this.manager = manager;
    }

    static MxModel load(String prefix, int epoch) {
        return load(prefix, epoch, null);
    }

    static MxModel load(String prefix, int epoch, Context context) {
        context = Context.defaultIfNull(context);
        MxNDManager subManager = MxNDManager.getSystemManager().newSubManager(context);
        Symbol symbol = Symbol.load(subManager, prefix + "-symbol.json");
        String paramFile = String.format("%s-%04d.params", prefix, epoch);
        Path modelDir = Paths.get(paramFile).toAbsolutePath().getParent();
        NDList paramNDlist = JnaUtils.loadNdArray(subManager, Paths.get(paramFile));

        List<Parameter> parameters = new ArrayList<>();
        for (Pair<String, NDArray> pair : paramNDlist) {
            String key = pair.getKey();
            if (key == null) {
                throw new IllegalArgumentException("Array names must be present in parameter file");
            }
            String paramName = key.split(":", 2)[1];
            NDArray array = pair.getValue().asInContext(context, true);
            parameters.add(new Parameter(paramName, array));
        }

        // TODO: Check if Symbol has all names that params file have
        return new MxModel(modelDir, new MxSymbolBlock(symbol, parameters, subManager), subManager);
    }

    /** {@inheritDoc} */
    @Override
    public <I, O> Predictor<I, O> newPredictor(Translator<I, O> translator) {
        return MxEngine.getInstance().newPredictor(this, translator, manager.getContext());
    }

    /** {@inheritDoc} */
    @Override
    public <I, O> Predictor<I, O> newPredictor(Translator<I, O> translator, Context context) {
        context = Context.defaultIfNull(context, manager.getContext());
        return MxEngine.getInstance().newPredictor(this, translator, context);
    }

    /** {@inheritDoc} */
    @Override
    public Model cast(DataType dataType) {
        Block newBlock = block.cast(dataType);
        return new MxModel(modelDir, newBlock, manager);
    }

    /** {@inheritDoc} */
    @Override
    public DataDesc[] describeInput() {
        if (inputData == null) {
            inputData = block.describeInput();
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
    public Block getBlock() {
        return block;
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getManager() {
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        manager.close();
    }
}
