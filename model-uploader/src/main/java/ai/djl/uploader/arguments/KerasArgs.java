/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.uploader.arguments;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

public class KerasArgs implements Arguments {

    private Path artifactPath;
    private String modelName;

    public void setArtifactPath(Path artifactPath) {
        this.artifactPath = artifactPath;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public Path getArtifactPath() {
        return artifactPath.resolve("keras").toAbsolutePath();
    }

    @Override
    public List<String> getArgs() {
        return Arrays.asList("--input_path", artifactPath.toString(), "--model_name", modelName);
    }
}
