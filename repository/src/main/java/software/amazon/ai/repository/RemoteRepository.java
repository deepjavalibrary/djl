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
import java.util.List;
import java.util.Map;

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
    public URI getBaseUri() {
        return uri;
    }

    @Override
    public Artifact resolve(MRL mrl, String version, Map<String, String> filter)
            throws IOException {
        URI mrlUri = mrl.toURI();
        URI base = uri.resolve(mrlUri.getPath());
        URI file = base.resolve("metadata.json");
        // TODO:
        // 1. check cache first
        // 2. check if offline mode or not
        // 3. download metadata if needed

        try (InputStream is = file.toURL().openStream();
                Reader reader = new InputStreamReader(is, StandardCharsets.UTF_8)) {
            Metadata metadata = GSON.fromJson(reader, Metadata.class);
            VersionRange range = VersionRange.parse(version);
            List<Artifact> artifacts = metadata.search(range, filter);
            if (artifacts.isEmpty()) {
                return null;
            }
            // TODO: find hightest version.
            return artifacts.get(0);
        }
    }
}
