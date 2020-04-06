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
import java.util.Collections;
import java.util.Set;
import software.amazon.awssdk.services.s3.S3Client;

/** A class responsible to create {@link S3Repository} instances. */
public class S3RepositoryFactory implements RepositoryFactory {

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
        String path = uri.getPath();
        if (!path.isEmpty()) {
            path = path.substring(1);
        }
        return new S3Repository(client, name, uri.getHost(), path);
    }

    /** {@inheritDoc} */
    @Override
    public Set<String> getSupportedScheme() {
        return Collections.singleton("s3");
    }
}
