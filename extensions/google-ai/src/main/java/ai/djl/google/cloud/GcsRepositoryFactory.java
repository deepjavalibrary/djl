/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.google.cloud;

import ai.djl.repository.Repository;
import ai.djl.repository.RepositoryFactory;

import com.google.cloud.storage.Storage;
import com.google.cloud.storage.StorageOptions;

import java.net.URI;
import java.util.Collections;
import java.util.Set;

/** A class responsible to create {@link GcsRepository} instances. */
public class GcsRepositoryFactory implements RepositoryFactory {

    private Storage storage;

    /** Creates a {@code GcsRepositoryFactory}. */
    public GcsRepositoryFactory() {}

    /**
     * Creates a {@code GcsRepositoryFactory} instance with the specified {@code Storage} client.
     *
     * @param storage the Google Cloud {@code Storage} client
     */
    public GcsRepositoryFactory(Storage storage) {
        this.storage = storage;
    }

    /** {@inheritDoc} */
    @Override
    public Repository newInstance(String name, URI uri) {
        String scheme = uri.getScheme();
        if (!"gs".equalsIgnoreCase(scheme)) {
            throw new IllegalArgumentException("Invalid gs url: " + uri);
        }

        if (storage == null) {
            storage = StorageOptions.getDefaultInstance().getService();
        }
        return new GcsRepository(name, uri, storage);
    }

    /** {@inheritDoc} */
    @Override
    public Set<String> getSupportedScheme() {
        return Collections.singleton("gs");
    }
}
