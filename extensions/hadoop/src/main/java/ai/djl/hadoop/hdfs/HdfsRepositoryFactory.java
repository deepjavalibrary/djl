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
package ai.djl.hadoop.hdfs;

import ai.djl.repository.Repository;
import ai.djl.repository.RepositoryFactory;
import java.net.URI;
import java.util.Collections;
import java.util.Set;
import org.apache.hadoop.conf.Configuration;

/** A class responsible to create {@link HdfsRepository} instances. */
public class HdfsRepositoryFactory implements RepositoryFactory {

    private Configuration config;

    /** Creates an {@code HdfsRepositoryFactory} instance with default {@code Configuration}. */
    public HdfsRepositoryFactory() {
        this(new Configuration());
    }

    /**
     * Creates an {@code HdfsRepositoryFactory} instance with the specified {@code Configuration}.
     *
     * @param config the {@code Configuration}
     */
    public HdfsRepositoryFactory(Configuration config) {
        this.config = config;
    }

    /** {@inheritDoc} */
    @Override
    public Repository newInstance(String name, URI uri) {
        String scheme = uri.getScheme();
        if (!"hdfs".equalsIgnoreCase(scheme)) {
            throw new IllegalArgumentException("Invalid hdfs url: " + uri);
        }
        return new HdfsRepository(name, uri, config);
    }

    /** {@inheritDoc} */
    @Override
    public Set<String> getSupportedScheme() {
        return Collections.singleton("hdfs");
    }
}
