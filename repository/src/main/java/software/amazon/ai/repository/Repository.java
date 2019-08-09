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
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipInputStream;

public interface Repository {

    Gson GSON =
            new GsonBuilder()
                    .setDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
                    .setPrettyPrinting()
                    .create();

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

    default InputStream openStream(Artifact.Item item, String path) throws IOException {
        Artifact artifact = item.getArtifact();
        URI artifactUri = artifact.getResourceUri();

        URI fileUri = URI.create(item.getUri());
        if (fileUri.isAbsolute()) {
            Path cacheDir = getCacheDirectory();
            Path resourceDir = cacheDir.resolve(artifactUri.getPath());
            String type = item.getType();
            String fileName = item.getName();
            Path cachedFile;
            if ("dir".equals(type)) {
                if (!fileName.isEmpty()) {
                    cachedFile = resourceDir.resolve(fileName);
                } else {
                    cachedFile = resourceDir;
                }
                cachedFile = cachedFile.resolve(path);
            } else {
                cachedFile = resourceDir.resolve(fileName);
            }
            return Files.newInputStream(cachedFile);
        }
        URI itemUri = getBaseUri().resolve(artifactUri.resolve(item.getUri()));
        return itemUri.toURL().openStream();
    }

    default void prepare(Artifact artifact) throws IOException {
        Path cacheDir = getCacheDirectory();
        URI resourceUri = artifact.getResourceUri();
        Path resourceDir = cacheDir.resolve(resourceUri.getPath());
        if (Files.exists(resourceDir)) {
            return;
        }

        // TODO: extract to temp directory first.
        Files.createDirectories(resourceDir);
        Metadata metadata = artifact.getMetadata();
        URI baseUri = metadata.getRepositoryUri();
        Map<String, Artifact.Item> files = artifact.getFiles();
        for (Map.Entry<String, Artifact.Item> entry : files.entrySet()) {
            Artifact.Item item = entry.getValue();
            URI fileUri = URI.create(item.getUri());
            if (!fileUri.isAbsolute()) {
                fileUri = getBaseUri().resolve(baseUri).resolve(fileUri);
            }

            // This is file is on remote, download it
            String fileName = item.getName();
            String extension = item.getExtension();
            if ("dir".equals(item.getType())) {
                Path dir;
                if (!fileName.isEmpty()) {
                    // honer the name set in metadata.json
                    dir = resourceDir.resolve(fileName);
                    Files.createDirectories(dir);
                } else {
                    dir = resourceDir;
                }
                if (!"zip".equals(extension)) {
                    throw new UnsupportedOperationException(
                            "File type is not supported: " + extension);
                }
                try (InputStream is = fileUri.toURL().openStream()) {
                    ZipUtils.unzip(is, dir);
                }
                return;
            }

            try (InputStream is = fileUri.toURL().openStream()) {
                Path file = resourceDir.resolve(fileName);
                if ("zip".equals(extension)) {
                    try (ZipInputStream zis = new ZipInputStream(is)) {
                        zis.getNextEntry();
                        Files.copy(zis, file);
                    }
                } else if ("gzip".equals(extension)) {
                    try (GZIPInputStream zis = new GZIPInputStream(is)) {
                        Files.copy(zis, file);
                    }
                } else if (extension.isEmpty()) {
                    Files.copy(is, file);
                } else {
                    throw new UnsupportedOperationException(
                            "File type is not supported: " + extension);
                }
            }
        }

        // TODO: clean up obsoleted files
    }

    default Path getCacheDirectory() throws IOException {
        String cacheDir = System.getProperty("JOULE_CACHE_DIR");
        if (cacheDir == null) {
            cacheDir = System.getenv("JOULE_CACHE_DIR");
            if (cacheDir == null) {
                String userHome = System.getProperty("user.home");
                cacheDir = userHome + "/.joule/cache";
            }
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
