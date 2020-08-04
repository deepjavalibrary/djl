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

import ai.djl.repository.FilenameUtils;
import ai.djl.repository.Repository;
import ai.djl.repository.RepositoryFactory;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.hadoop.conf.Configuration;

/** A class responsible to create {@link HdfsRepository} instances. */
public class HdfsRepositoryFactory implements RepositoryFactory {

    private static final Pattern NAME_PATTERN = Pattern.compile("model_name=([^&]*)");
    private static final Pattern ARTIFACT_PATTERN = Pattern.compile("artifact_id=([^&]*)");

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
    public Repository newInstance(String name, String url) {
        URI uri = URI.create(url);
        String scheme = uri.getScheme();
        if (!"hdfs".equalsIgnoreCase(scheme)) {
            throw new IllegalArgumentException("Invalid hdfs url: " + url);
        }

        String path = uri.getPath();
        String fileName = Paths.get(path).toFile().getName();
        boolean isDirectory = !FilenameUtils.isArchiveFile(fileName);
        if (!isDirectory) {
            fileName = FilenameUtils.getNamePart(fileName);
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
            artifactId = fileName;
        }
        if (modelName == null) {
            modelName = artifactId;
        }
        if (path.isEmpty()) {
            path = "/";
        }
        try {
            uri =
                    new URI(
                            "hdfs",
                            uri.getUserInfo(),
                            uri.getHost(),
                            uri.getPort(),
                            null,
                            null,
                            null);
        } catch (URISyntaxException e) {
            throw new AssertionError(e);
        }

        return new HdfsRepository(config, name, uri, path, artifactId, modelName, isDirectory);
    }

    /** {@inheritDoc} */
    @Override
    public Set<String> getSupportedScheme() {
        return Collections.singleton("hdfs");
    }
}
