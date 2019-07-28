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
package software.amazon.ai.zoo;

import java.io.InputStream;
import java.net.URL;
import java.util.function.Function;
import software.amazon.ai.Model;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;

public class PretrainedModel implements Model {

    private Pretrained pretrained;

    public PretrainedModel(Pretrained pretrained) {
        this.pretrained = pretrained;
    }

    @Override
    public DataDesc[] describeInput() {
        return pretrained.describeInput();
    }

    @Override
    public DataDesc[] describeOutput() {
        return pretrained.describeOutput();
    }

    @Override
    public String[] getArtifactNames() {
        return new String[0];
    }

    @Override
    public <T> T getArtifact(String name, Function<InputStream, T> function) {
        return null;
    }

    @Override
    public URL getArtifact(String name) {
        return null;
    }

    @Override
    public InputStream getArtifactAsStream(String name) {
        return null;
    }

    @Override
    public Model cast(DataType dataType) {
        return null;
    }

    @Override
    public void close() {}
}
