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

import ai.djl.Application;
import ai.djl.util.Progress;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * {@code Repository} is a format for storing data {@link Artifact}s for various uses including deep
 * learning models and datasets.
 *
 * <p>This repository format is based off of the design of the Maven Repository format (See <a
 * href="https://maven.apache.org/guides/introduction/introduction-to-repositories.html">maven</a>).
 * Unlike in Maven, the data doesn't need to be located within the repository. Instead, the
 * repository only stores metadata including the URL and checksum of the actual data. When the
 * artifact is prepared, the data is downloaded, checked, and then stored in the {@code
 * ~/.djo-ai/cache} folder.
 *
 * <p>The artifacts are first divided into a number of {@link Metadata} files that can each have
 * multiple artifacts. The metadata files are identified by an {@link MRL} which contains:
 *
 * <ul>
 *   <li>type - The resource type, e.g. model or dataset.
 *   <li>Application - The resource application (See {@link Application}).
 *   <li>Group Id - The group id identifies the group publishing the artifacts using a reverse
 *       domain name system.
 *   <li>Artifact Id - The artifact id identifies the different artifacts published by a single
 *       group.
 * </ul>
 *
 * <p>Within each metadata are a number of artifacts that share the same groupId, artifactId, name,
 * description, website, and update date. The artifacts within the metadata differ primarily based
 * on name and properties. Note that there is a metadata name and a separate artifact name. The
 * properties are a map with string property names and string property values that can be used to
 * represent key differentiators between artifacts such as dataset, flavors, and image sizes. For
 * example, you might have a ResNet metadata file with different artifacts to represent different
 * hyperparameters and datasets used for training the ResNet.
 *
 * <p>Each artifact contains a {@link Version} number (which can be a snapshot version). The data in
 * the artifacts are represented by files in the format of an {@link Artifact.Item} and a parsed
 * JSON object of arguments. The files can either by a single file, an automatically extracted gzip
 * file, or an automatically extracted zip file that will be treated as a directory. These can be
 * used to store data such as the dataset, model parameters, and synset files. The arguments can be
 * used to store data about the model used for initialization. For example, it can store the image
 * size which can be used by the model loader for both initializing the block and setting up
 * resizing in the translator.
 *
 * <p>There are three kinds of repositories: a {@link LocalRepository}, {@link RemoteRepository},
 * and {@link SimpleRepository}. For all three kinds, new repositories should be created by calling
 * {@link Repository#newInstance(String, String)} with the location of the repository.
 */
public interface Repository {

    /**
     * Creates a new instance of a repository with a name and url.
     *
     * @param name the repository name
     * @param path the repository location
     * @return the new repository
     */
    static Repository newInstance(String name, Path path) {
        return RepositoryFactoryImpl.getFactory().newInstance(name, path.toUri().toString());
    }

    /**
     * Creates a new instance of a repository with a name and url.
     *
     * @param name the repository name
     * @param url the repository location
     * @return the new repository
     */
    static Repository newInstance(String name, String url) {
        return RepositoryFactoryImpl.getFactory().newInstance(name, url);
    }

    /**
     * Registers a {@link RepositoryFactory} to handle the specified url scheme.
     *
     * @param factory the {@link RepositoryFactory} to be registered
     */
    static void registerRepositoryFactory(RepositoryFactory factory) {
        RepositoryFactoryImpl.registerRepositoryFactory(factory);
    }

    /**
     * Creates a model {@code MRL} with specified application.
     *
     * @param application the desired application
     * @param groupId the desired groupId
     * @param artifactId the desired artifactId
     * @return a model {@code MRL}
     */
    default MRL model(Application application, String groupId, String artifactId) {
        return model(application, groupId, artifactId, null);
    }

    /**
     * Creates a model {@code MRL} with specified application.
     *
     * @param application the desired application
     * @param groupId the desired groupId
     * @param artifactId the desired artifactId
     * @param version the resource version
     * @return a model {@code MRL}
     */
    default MRL model(Application application, String groupId, String artifactId, String version) {
        return MRL.model(this, application, groupId, artifactId, version);
    }

    /**
     * Creates a dataset {@code MRL} with specified application.
     *
     * @param application the desired application
     * @param groupId the desired groupId
     * @param artifactId the desired artifactId
     * @return a dataset {@code MRL}
     */
    default MRL dataset(Application application, String groupId, String artifactId) {
        return dataset(application, groupId, artifactId, null);
    }

    /**
     * Creates a dataset {@code MRL} with specified application.
     *
     * @param application the desired application
     * @param groupId the desired groupId
     * @param artifactId the desired artifactId
     * @param version the resource version
     * @return a dataset {@code MRL}
     */
    default MRL dataset(
            Application application, String groupId, String artifactId, String version) {
        return MRL.dataset(this, application, groupId, artifactId, version);
    }

    /**
     * Returns whether the repository is remote repository.
     *
     * @return whether the repository is remote repository
     */
    boolean isRemote();

    /**
     * Returns the repository name.
     *
     * @return the repository name
     */
    String getName();

    /**
     * Returns the URI to the base of the repository.
     *
     * @return the URI
     */
    URI getBaseUri();

    /**
     * Returns the metadata at a mrl.
     *
     * @param mrl the mrl of the metadata to retrieve
     * @return the metadata
     * @throws IOException if it failed to load the metadata
     */
    Metadata locate(MRL mrl) throws IOException;

    /**
     * Returns the artifact matching a mrl, version, and property filter.
     *
     * @param mrl the mrl to match the artifact against
     * @param filter the property filter
     * @return the matched artifact
     * @throws IOException if it failed to load the artifact
     */
    Artifact resolve(MRL mrl, Map<String, String> filter) throws IOException;

    /**
     * Returns an {@link InputStream} for an item in a repository.
     *
     * @param item the item to open
     * @param path the path to a file if the item is a zipped directory. Otherwise, pass null
     * @return the file stream
     * @throws IOException if it failed to open the stream
     */
    InputStream openStream(Artifact.Item item, String path) throws IOException;

    /**
     * Returns the path to a file for the item.
     *
     * @param item the item to find the path for
     * @param path the path to a file if the item is a zipped directory. Otherwise, pass null
     * @return the file path
     * @throws IOException if it failed to find the path
     */
    Path getFile(Artifact.Item item, String path) throws IOException;

    /**
     * Returns the list of files directly within a specified directory in a zipped directory item.
     *
     * @param item the zipped directory item
     * @param path the path within the zip directory
     * @return the list of files/directories
     * @throws IOException if it failed to list the directory
     */
    String[] listDirectory(Artifact.Item item, String path) throws IOException;

    /**
     * Prepares the artifact for use.
     *
     * @param artifact the artifact to prepare
     * @throws IOException if it failed to prepare
     */
    default void prepare(Artifact artifact) throws IOException {
        prepare(artifact, null);
    }

    /**
     * Prepares the artifact for use with progress tracking.
     *
     * @param artifact the artifact to prepare
     * @param progress the progress tracker
     * @throws IOException if it failed to prepare
     */
    void prepare(Artifact artifact, Progress progress) throws IOException;

    /**
     * Returns the cache directory for the repository.
     *
     * @return the cache directory path
     * @throws IOException if it failed to ensure the creation of the cache directory
     */
    Path getCacheDirectory() throws IOException;

    /**
     * Returns the resource directory for the an artifact.
     *
     * @param artifact the artifact whose resource directory to return
     * @return the resource directory path
     * @throws IOException if it failed to ensure the creation of the cache directory
     */
    default Path getResourceDirectory(Artifact artifact) throws IOException {
        return getCacheDirectory().resolve(artifact.getResourceUri().getPath());
    }

    /**
     * Returns a list of {@link MRL}s in the repository.
     *
     * <p>An empty list will be returned if underlying {@code Repository} implementation does not
     * support this feature.
     *
     * @return a list of {@link MRL}s in the repository
     */
    List<MRL> getResources();
}
