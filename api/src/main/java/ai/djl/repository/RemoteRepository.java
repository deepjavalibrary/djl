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
package ai.djl.repository;

import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;
import java.io.IOException;
import java.io.InputStream;
import java.io.Reader;
import java.io.Writer;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Date;
import java.util.List;
import java.util.Map;

/**
 * A {@code RemoteRepository} is a {@link Repository} located on a remote web server.
 *
 * @see Repository
 */
public class RemoteRepository extends AbstractRepository {

    private static final long ONE_DAY = Duration.ofDays(1).toMillis();

    private String name;
    private URI uri;

    /**
     * (Internal) Constructs a remote repository.
     *
     * <p>Use {@link Repository#newInstance(String, String)}.
     *
     * @param name the repository name
     * @param uri the repository location
     */
    protected RemoteRepository(String name, URI uri) {
        this.name = name;
        this.uri = uri;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isRemote() {
        return true;
    }

    /** {@inheritDoc} */
    @Override
    public String getName() {
        return name;
    }

    /** {@inheritDoc} */
    @Override
    public URI getBaseUri() {
        return uri;
    }

    /** {@inheritDoc} */
    @Override
    public Metadata locate(MRL mrl) throws IOException {
        URI mrlUri = mrl.toURI();
        URI file = uri.resolve(mrlUri.getPath() + "/metadata.json");
        Path cacheDir = getCacheDirectory().resolve(mrlUri.getPath());
        if (!Files.exists(cacheDir)) {
            Files.createDirectories(cacheDir);
        }
        Path cacheFile = cacheDir.resolve("metadata.json");
        if (Files.exists(cacheFile)) {
            try (Reader reader = Files.newBufferedReader(cacheFile)) {
                Metadata metadata = JsonUtils.GSON_PRETTY.fromJson(reader, Metadata.class);
                metadata.init();
                Date lastUpdated = metadata.getLastUpdated();
                if (Boolean.getBoolean("offline")
                        || System.currentTimeMillis() - lastUpdated.getTime() < ONE_DAY) {
                    metadata.setRepositoryUri(mrlUri);
                    return metadata;
                }
            }
        }

        Path tmp = Files.createTempFile(cacheDir, "metadata", ".tmp");
        try (InputStream is = file.toURL().openStream()) {
            String json = Utils.toString(is);
            Metadata metadata = JsonUtils.GSON_PRETTY.fromJson(json, Metadata.class);
            metadata.init();
            metadata.setLastUpdated(new Date());
            try (Writer writer = Files.newBufferedWriter(tmp)) {
                writer.write(JsonUtils.GSON_PRETTY.toJson(metadata));
            }
            Utils.moveQuietly(tmp, cacheFile);
            metadata.setRepositoryUri(mrlUri);
            return metadata;
        } finally {
            Utils.deleteQuietly(tmp);
        }
    }

    /** {@inheritDoc} */
    @Override
    public Artifact resolve(MRL mrl, Map<String, String> filter) throws IOException {
        Metadata metadata = locate(mrl);
        VersionRange range = VersionRange.parse(mrl.getVersion());
        List<Artifact> artifacts = metadata.search(range, filter);
        if (artifacts.isEmpty()) {
            return null;
        }
        // TODO: find highest version.
        return artifacts.get(0);
    }
}
