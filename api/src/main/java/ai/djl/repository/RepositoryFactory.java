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
package ai.djl.repository;

import java.net.URI;
import java.util.Set;

/** A interface responsible to create {@link ai.djl.repository.Repository} instances. */
public interface RepositoryFactory {

    /**
     * Creates a new instance of a repository with a name and url.
     *
     * @param name the repository name
     * @param uri the repository location
     * @return the new repository
     */
    Repository newInstance(String name, URI uri);

    /**
     * Returns a set of URI scheme that the {@code RepositoryFactory} supports.
     *
     * @return a set of URI scheme that the {@code RepositoryFactory} supports
     */
    Set<String> getSupportedScheme();
}
