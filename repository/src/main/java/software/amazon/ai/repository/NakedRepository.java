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
package software.amazon.ai.repository;

import java.net.URI;
import java.nio.file.Path;
import java.util.Map;

public class NakedRepository implements Repository {

    private String name;
    private Path path;

    public NakedRepository(Path path) {
        this(path.toFile().getName(), path);
    }

    public NakedRepository(String name, Path path) {
        this.name = name;
        this.path = path;
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public URI getBaseUri() {
        return path.toUri();
    }

    @Override
    public Artifact resolve(MRL mrl, String version, Map<String, String> filter) {
        Artifact artifact = new Artifact();
        artifact.setGroupId(mrl.getGroupId());
        artifact.setArtifactId(mrl.getArtifactId());
        artifact.setVersion(version);
        artifact.setBaseUri(path.toUri());
        return artifact;
    }

    @Override
    public void prepare(Artifact artifact) {
        // Do nothing
    }
}
