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
import software.amazon.ai.Block;
import software.amazon.ai.Context;
import software.amazon.ai.Model;
import software.amazon.ai.Parameter;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;

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

    static MxModel loadModel(String prefix, int epoch) {
        return loadModel(MxNDManager.getSystemManager(), prefix, epoch, null);
    }

    static MxModel loadModel(String prefix, int epoch, Context context) {
        return loadModel(MxNDManager.getSystemManager(), prefix, epoch, context);
    }

    @SuppressWarnings("unused")
    static MxModel loadModel(MxNDManager manager, String prefix, int epoch, Context context) {
        MxNDManager subManager = manager.newSubManager();
        Symbol symbol = Symbol.load(subManager, prefix + "-symbol.json");
        String paramFile = String.format("%s-%04d.params", prefix, epoch);
        Path modelDir = Paths.get(paramFile).toAbsolutePath().getParent();

        PointerByReference namesRef = new PointerByReference();

        // TODO: MXNet engine to support load NDArray on specific context.
        Pointer[] handles = JnaUtils.loadNdArray(paramFile, namesRef);
        String[] names = namesRef.getValue().getStringArray(0, handles.length);

        List<Parameter> parameters = new ArrayList<>();

        for (int i = 0; i < names.length; ++i) {
            String[] pair = names[i].split(":", 2);
            MxNDArray array = subManager.create(handles[i]);
            if (pair.length == 2) {
                parameters.add(new Parameter(pair[1], array));
            } else {
                parameters.add(new Parameter(pair[0], array));
            }
        }
        // TODO: Check if Symbol has all names that params file have
        return new MxModel(modelDir, new SymbolBlock(symbol, parameters, subManager), subManager);
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

    @Override
    public Block getBlock() {
        return block;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        manager.close();
    }
}
