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

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

public interface Repository {

    Gson GSON = new GsonBuilder().setDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'").create();

    static Repository newInstance(String name, String url) {
        URI uri = URI.create(url);
        if (!uri.isAbsolute()) {
            return new LocalRepository(name, Paths.get(url));
        }

        String scheme = uri.getScheme();
        if ("file".equalsIgnoreCase(scheme)) {
            return new LocalRepository(name, Paths.get(uri.getPath()));
        }

        return new RemoteRepository(name, uri);
    }

    String getName();

    URI getBaseUri();

    Metadata locate(MRL mrl) throws IOException;

    Artifact resolve(MRL mrl, String version, Map<String, String> filter) throws IOException;

    default InputStream openStream(Artifact.Item item) throws IOException {
        Artifact artifact = item.getArtifact();
        URI artifactUri = artifact.getResourceUri();

        URI fileUri = URI.create(item.getUri());
        if (fileUri.isAbsolute()) {
            Path cacheDir = getCacheDirectory();
            Path resourceDir = cacheDir.resolve(artifactUri.getPath());
            Path cachedFile = resourceDir.resolve(item.getSha1Hash());
            return Files.newInputStream(cachedFile);
        }
        URI itemUri = getBaseUri().resolve(artifactUri.resolve(item.getUri()));
        return itemUri.toURL().openStream();
    }

    default void prepare(Artifact artifact) throws IOException {
        Path cacheDir = getCacheDirectory();
        URI resourceUri = artifact.getResourceUri();
        Path resourceDir = cacheDir.resolve(resourceUri.getPath());
        if (!Files.exists(resourceDir)) {
            Files.createDirectories(resourceDir);
        }

        Map<String, Artifact.Item> files = artifact.getFiles();
        for (Map.Entry<String, Artifact.Item> entry : files.entrySet()) {
            Artifact.Item item = entry.getValue();
            URI fileUri = URI.create(item.getUri());
            if (fileUri.isAbsolute()) {
                Path cachedFile = resourceDir.resolve(item.getSha1Hash());
                if (Files.exists(cachedFile)) {
                    continue;
                }
                try (InputStream is = fileUri.toURL().openStream()) {
                    // TODO: save to tmp file first.
                    Files.copy(is, cachedFile);
                }
            }
        }

        // TODO: clean up obsoleted files
    }

    default Path getCacheDirectory() throws IOException {
        String cacheDir = System.getProperty("JOULE_CACHE_DIR");
        if (cacheDir == null) {
            String userHome = System.getProperty("user.home");
            cacheDir = userHome + "/.joule/cache";
        }

        Path dir = Paths.get(cacheDir, "repo");
        if (Files.notExists(dir)) {
            Files.createDirectories(dir);
        } else if (!Files.isDirectory(dir)) {
            throw new IOException("Failed initialize cache directory: " + dir.toString());
        }
        return dir;
    }
}
