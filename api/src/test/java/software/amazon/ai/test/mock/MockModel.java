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
package software.amazon.ai.test.mock;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import software.amazon.ai.Context;
import software.amazon.ai.Model;
import software.amazon.ai.inference.Predictor;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.nn.Block;
import software.amazon.ai.training.Trainer;
import software.amazon.ai.training.initializer.Initializer;
import software.amazon.ai.training.optimizer.Optimizer;
import software.amazon.ai.translate.TrainTranslator;
import software.amazon.ai.translate.Translator;

public class MockModel implements Model {

    private Map<String, Object> artifacts = new ConcurrentHashMap<>();

    @Override
    public <I, L, O> Trainer<I, L, O> newTrainer(TrainTranslator<I, L, O> trainTranslator) {
        return null;
    }

    @Override
    public <I, L, O> Trainer<I, L, O> newTrainer(
            TrainTranslator<I, L, O> trainTranslator, Optimizer optimizer) {
        return null;
    }

    @Override
    public <I, L, O> Trainer<I, L, O> newTrainer(
            TrainTranslator<I, L, O> trainTranslator, Optimizer optimizer, Context context) {
        return null;
    }

    @Override
    public <I, O> Predictor<I, O> newPredictor(Translator<I, O> translator, Context context) {
        return new MockPredictor<>(this, translator, context);
    }

    @Override
    public void setInitializer(Initializer initializer) {}

    @Override
    public void setInitializer(Initializer initializer, boolean overwrite) {}

    @Override
    public DataDesc[] describeInput() {
        return new DataDesc[0];
    }

    @Override
    public DataDesc[] describeOutput() {
        return new DataDesc[0];
    }

    @Override
    public String[] getArtifactNames() {
        return new String[] {"synset.txt"};
    }

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

    @Override
    public URL getArtifact(String name) {
        return Thread.currentThread().getContextClassLoader().getResource(name);
    }

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
        return null;
    }

    @Override
    public NDManager getManager() {
        return null;
    }

    @Override
    public Model cast(DataType dataType) {
        return null;
    }

    @Override
    public void close() {}

    public void setArtifacts(Map<String, Object> artifacts) {
        this.artifacts = artifacts;
    }
}
