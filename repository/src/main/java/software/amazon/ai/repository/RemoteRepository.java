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

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URI;
import java.nio.charset.StandardCharsets;

public class RemoteRepository implements Repository {

    private String name;
    private URI uri;

    public RemoteRepository(String name, URI uri) {
        this.name = name;
        this.uri = uri;
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public Artifact resolve(Anchor anchor) throws IOException {
        // TODO:
        // 1. check cache first
        // 2. check if offline mode or not
        // 3. download metadata if needed
        URI base = uri.resolve(anchor.getBaseUri());
        URI file = base.resolve("metadata.json");
        Version version;
        try (InputStream is = file.toURL().openStream();
                Reader reader = new InputStreamReader(is, StandardCharsets.UTF_8)) {
            Metadata metadata = GSON.fromJson(reader, Metadata.class);
            VersionRange range = VersionRange.parse(anchor.getVersion());
            version = metadata.resolve(range);
            if (version == null) {
                return null;
            }
        } catch (IOException e) {
            return null;
        }

        URI path = base.resolve(version.toString()).resolve("artifact.json");
        try (InputStream is = file.toURL().openStream();
                Reader reader = new InputStreamReader(is, StandardCharsets.UTF_8)) {
            Artifact artifact = GSON.fromJson(reader, Artifact.class);
            artifact.setBaseUri(path);
            return artifact;
        }
    }

    @Override
    public void prepare(Artifact artifact) throws IOException {
        // TODO:
        // 1. check cache if need download.
        // 2. Check offline mode or not
        // 3. Download from remote and set lastUpdated in metadata.
        // 4. Unzip or some process if needed by configuration.
    }
}
