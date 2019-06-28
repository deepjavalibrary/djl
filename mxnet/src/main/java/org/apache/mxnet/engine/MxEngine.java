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
import java.lang.management.MemoryUsage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.nn.MxNNIndex;
import software.amazon.ai.Context;
import software.amazon.ai.Model;
import software.amazon.ai.Profiler;
import software.amazon.ai.Translator;
import software.amazon.ai.engine.Engine;
import software.amazon.ai.inference.Predictor;
import software.amazon.ai.nn.NNIndex;
import software.amazon.ai.training.Trainer;

public class MxEngine extends Engine {

    public static final NNIndex NN_INDEX = new MxNNIndex();

    MxEngine() {}

    /** {@inheritDoc} */
    @Override
    public String getEngineName() {
        return "MXNet";
    }

    /** {@inheritDoc} */
    @Override
    public int getGpuCount() {
        return JnaUtils.getGpuCount();
    }

    /** {@inheritDoc} */
    @Override
    public MemoryUsage getGpuMemory(Context context) {
        long[] mem = JnaUtils.getGpuMemory(context);
        long committed = mem[1] - mem[0];
        return new MemoryUsage(-1, committed, committed, mem[1]);
    }

    /** {@inheritDoc} */
    @Override
    public Context defaultContext() {
        if (getGpuCount() > 0) {
            return Context.gpu(0);
        }
        return Context.cpu();
    }

    /** {@inheritDoc} */
    @Override
    public String getVersion() {
        int version = JnaUtils.getVersion();
        int major = version / 10000;
        int minor = version / 100 - major * 100;
        int patch = version % 100;

        return major + "." + minor + '.' + patch;
    }

    /**
     * Load the MXNet model from specified location.
     *
     * <p>MXNet engine looks for modelName.json and modelName-xxxx.params files in specified
     * directory. By default, MXNet engine will pick up latest epoch of parameter file. However,
     * user can explicitly an epoch to be loaded:
     *
     * <pre>
     * Map&lt;String, String&gt; options = new HashMap&lt;&gt;()
     * <b>options.put("epoch", "3");</b>
     * Model model = Model.load(modelPath, "squeezenet", options);
     * </pre>
     *
     * @param modelPath Directory of the model
     * @param modelName Name/Prefix of the model
     * @param options load model options, check document for specific engine
     * @return {@link Model} contains the model information
     * @throws IOException Exception for file loading
     */
    @Override
    public Model loadModel(Path modelPath, String modelName, Map<String, String> options)
            throws IOException {
        Path modelDir;
        if (Files.isDirectory(modelPath)) {
            modelDir = modelPath;
        } else {
            modelDir = modelPath.toAbsolutePath().getParent();
            if (modelDir == null) {
                throw new AssertionError("Invalid path: " + modelPath.toString());
            }
        }
        String modelPrefix = modelDir.resolve(modelName).toAbsolutePath().toString();

        String epochOption = null;
        if (options != null) {
            epochOption = options.get("epoch");
        }
        int epoch;
        if (epochOption == null) {
            final Pattern pattern = Pattern.compile(Pattern.quote(modelName) + "-(\\d{4}).params");
            List<Integer> checkpoints =
                    Files.walk(modelDir, 1)
                            .map(
                                    p -> {
                                        Matcher m = pattern.matcher(p.toFile().getName());
                                        if (m.matches()) {
                                            return Integer.parseInt(m.group(1));
                                        }
                                        return null;
                                    })
                            .filter(Objects::nonNull)
                            .sorted()
                            .collect(Collectors.toList());
            if (checkpoints.isEmpty()) {
                throw new IOException("Parameter files not found: " + modelPrefix + "-0001.params");
            }
            epoch = checkpoints.get(checkpoints.size() - 1);
        } else {
            epoch = Integer.parseInt(epochOption);
        }

        return MxModel.loadModel(modelPrefix, epoch);
    }

    /** {@inheritDoc} */
    @Override
    public NNIndex getNNIndex() {
        return NN_INDEX;
    }

    /** {@inheritDoc} */
    @Override
    public <I, O> Predictor<I, O> newPredictor(
            Model model, Translator<I, O> translator, Context context) {
        return new MxPredictor<>((MxModel) model, translator, context);
    }

    /** {@inheritDoc} */
    @Override
    public Trainer newTrainer(Model model, Context context) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void setProfiler(Profiler profiler) {}
}
