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
package ai.djl.aws.s3;

import ai.djl.repository.Repository;
import ai.djl.repository.RepositoryFactory;
import java.net.URI;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import software.amazon.awssdk.services.s3.S3Client;

/** A class responsible to create {@link S3Repository} instances. */
public class S3RepositoryFactory implements RepositoryFactory {

    private static final Pattern NAME_PATTERN = Pattern.compile("model_name=([^&]*)");
    private static final Pattern ARTIFACT_PATTERN = Pattern.compile("artifact_id=([^&]*)");

    private S3Client client;

    /** Creates an {@code S3RepositoryFactory} instance with default {@code S3Client}. */
    public S3RepositoryFactory() {
        this(S3Client.builder().build());
    }

    /**
     * Creates an {@code S3RepositoryFactory} instance with the specified {@code S3Client}.
     *
     * @param client the {@code S3Client}
     */
    public S3RepositoryFactory(S3Client client) {
        this.client = client;
    }

    /** {@inheritDoc} */
    @Override
    public Repository newInstance(String name, String url) {
        URI uri = URI.create(url);
        String scheme = uri.getScheme();
        if (!"s3".equalsIgnoreCase(scheme)) {
            throw new IllegalArgumentException("Invalid s3 url: " + url);
        }

        String bucket = uri.getHost();
        String prefix = uri.getPath();
        if (!prefix.isEmpty()) {
            prefix = prefix.substring(1);
        }
        if (!prefix.isEmpty() && !prefix.endsWith("/")) {
            prefix += '/'; // NOPMD
        }

        String modelName = null;
        String artifactId = null;
        String query = uri.getQuery();
        if (query != null) {
            Matcher matcher = NAME_PATTERN.matcher(query);
            if (matcher.find()) {
                modelName = matcher.group(1);
            }
            matcher = ARTIFACT_PATTERN.matcher(query);
            if (matcher.find()) {
                artifactId = matcher.group(1);
            }
        }

        if (artifactId == null) {
            Path path = Paths.get(prefix);
            if (path.getNameCount() == 0) {
                artifactId = bucket;
            } else {
                Path fileName = path.getFileName();
                if (fileName == null) {
                    throw new AssertionError("This should never happen.");
                }
                artifactId = fileName.toString();
            }
        }
        if (modelName == null) {
            modelName = artifactId;
        }
        return new S3Repository(client, name, bucket, prefix, artifactId, modelName);
    }

    /** {@inheritDoc} */
    @Override
    public Set<String> getSupportedScheme() {
        return Collections.singleton("s3");
    }
}
