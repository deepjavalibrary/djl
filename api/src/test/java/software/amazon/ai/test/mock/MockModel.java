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
import software.amazon.ai.Model;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;

public class MockModel implements Model {

    private Map<String, Object> artifacts = new ConcurrentHashMap<>();

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
    public Model cast(DataType dataType) {
        return null;
    }

    @Override
    public void close() {}

    public void setArtifacts(Map<String, Object> artifacts) {
        this.artifacts = artifacts;
    }
}
