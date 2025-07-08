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

import ai.djl.repository.zoo.ModelLoader;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.util.ClassLoaderUtils;
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;

import com.google.gson.JsonParseException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.Reader;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.Locale;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

class RepositoryFactoryImpl implements RepositoryFactory {

    private static final Logger logger = LoggerFactory.getLogger(RepositoryFactoryImpl.class);

    private static final RepositoryFactory FACTORY = new RepositoryFactoryImpl();
    private static final Map<String, RepositoryFactory> REGISTRY = init();
    private static final Pattern PATTERN = Pattern.compile("(.+)/([\\d.]+)(/(.*))?");

    static RepositoryFactory getFactory() {
        return FACTORY;
    }

    /** {@inheritDoc} */
    @Override
    public Repository newInstance(String name, URI uri) {
        String scheme = uri.getScheme();
        if (scheme == null) {
            scheme = "file";
        }
        RepositoryFactory factory = REGISTRY.get(scheme);
        if (factory != null) {
            return factory.newInstance(name, uri);
        }

        try {
            uri.toURL();
        } catch (MalformedURLException e) {
            throw new IllegalArgumentException("Malformed URL: " + uri, e);
        }

        if ("tfhub.dev".equals(uri.getHost().toLowerCase(Locale.ROOT))) {
            // Handle tfhub case
            String path = uri.getPath();
            if (path.endsWith("/")) {
                path = path.substring(0, path.length() - 1);
            }
            path = "/tfhub-modules" + path + ".tar.gz";
            try {
                uri = new URI("https", null, "storage.googleapis.com", -1, path, null, null);
            } catch (URISyntaxException e) {
                throw new IllegalArgumentException("Failed to append query string: " + uri, e);
            }
            String[] tokens = path.split("/");
            String modelName = tokens[tokens.length - 2];
            return new SimpleUrlRepository(name, uri, modelName);
        }

        Path path = Paths.get(parseFilePath(uri));
        String fileName = path.toFile().getName();
        if (FilenameUtils.isArchiveFile(fileName)) {
            fileName = FilenameUtils.getNamePart(fileName);
            return new SimpleUrlRepository(name, uri, fileName);
        }
        return new RpcRepository(name, uri);
    }

    /** {@inheritDoc} */
    @Override
    public Set<String> getSupportedScheme() {
        return REGISTRY.keySet();
    }

    static void registerRepositoryFactory(RepositoryFactory factory) {
        for (String scheme : factory.getSupportedScheme()) {
            REGISTRY.put(scheme, factory);
        }
    }

    private static Map<String, RepositoryFactory> init() {
        Map<String, RepositoryFactory> registry = new ConcurrentHashMap<>();
        registry.put("file", new LocalRepositoryFactory());
        registry.put("jar", new JarRepositoryFactory());
        registry.put("djl", new DjlRepositoryFactory());
        if (S3RepositoryFactory.findS3Fuse() != null) {
            registry.put("s3", new S3RepositoryFactory());
        }
        if (GcsRepositoryFactory.findGcsFuse() != null) {
            registry.put("gs", new GcsRepositoryFactory());
        }

        ServiceLoader<RepositoryFactory> factories = ServiceLoader.load(RepositoryFactory.class);
        for (RepositoryFactory factory : factories) {
            for (String scheme : factory.getSupportedScheme()) {
                registry.put(scheme, factory);
            }
        }
        return registry;
    }

    static String parseFilePath(URI uri) {
        String uriPath = uri.getPath();
        if (uriPath == null) {
            uriPath = uri.getSchemeSpecificPart();
        }
        if (uriPath.startsWith("file:")) {
            // handle jar:file:/ url
            uriPath = uriPath.substring(5);
        }
        if (uriPath.startsWith("/") && System.getProperty("os.name").startsWith("Win")) {
            uriPath = uriPath.substring(1);
        }
        return uriPath;
    }

    private static String exec(String... cmd) throws IOException, InterruptedException {
        Process exec = new ProcessBuilder(cmd).redirectErrorStream(true).start();
        String logOutput;
        try (InputStream is = exec.getInputStream()) {
            logOutput = Utils.toString(is);
        }
        int exitCode = exec.waitFor();
        if (0 != exitCode) {
            logger.error("exit: {}, {}", exitCode, logOutput);
            throw new IOException("Failed to execute: [" + String.join(" ", cmd) + "]");
        } else {
            logger.debug("{}", logOutput);
        }
        return logOutput;
    }

    private static boolean isMounted(String path) throws IOException, InterruptedException {
        String out = exec("df");
        String[] lines = out.split("\\s");
        for (String line : lines) {
            if (line.trim().equals(path)) {
                logger.debug("Mount point already mounted");
                return true;
            }
        }
        return false;
    }

    private static final class JarRepositoryFactory implements RepositoryFactory {

        /** {@inheritDoc} */
        @Override
        public Repository newInstance(String name, URI uri) {
            String p = uri.getPath();
            if (p.startsWith("/")) {
                p = p.substring(1);
            }
            URL u = ClassLoaderUtils.getContextClassLoader().getResource(p);
            if (u == null) {
                throw new IllegalArgumentException("Resource not found: " + uri);
            }

            URI realUri;
            try {
                // resolve real uri: jar:file:/path/my_lib.jar!/model.zip
                realUri = u.toURI();
            } catch (URISyntaxException e) {
                throw new IllegalArgumentException("Resource not found: " + uri, e);
            }

            Path path = Paths.get(parseFilePath(realUri));
            String fileName = path.toFile().getName();
            if (FilenameUtils.isArchiveFile(fileName)) {
                fileName = FilenameUtils.getNamePart(fileName);
            }

            return new JarRepository(name, uri, fileName, realUri);
        }

        /** {@inheritDoc} */
        @Override
        public Set<String> getSupportedScheme() {
            return Collections.singleton("jar");
        }
    }

    private static final class LocalRepositoryFactory implements RepositoryFactory {

        /** {@inheritDoc} */
        @Override
        public Repository newInstance(String name, URI uri) {
            Path path = Paths.get(parseFilePath(uri));
            if (Files.exists(path) && Files.isDirectory(path)) {
                try {
                    if (Files.walk(path).anyMatch(f -> isLocalRepository(path, f))) {
                        logger.debug("Found local repository: {}", path);
                        return new LocalRepository(name, path.toUri(), path);
                    }
                } catch (IOException e) {
                    logger.warn("Failed locate metadata.json file, defaulting to simple", e);
                }
            }
            return new SimpleRepository(name, uri, path);
        }

        private boolean isLocalRepository(Path root, Path file) {
            if (!Files.isRegularFile(file) || root.equals(file.getParent())) {
                return false;
            }
            if (!"metadata.json".equals(file.toFile().getName())) {
                return false;
            }
            try (Reader reader = Files.newBufferedReader(file)) {
                Metadata metadata = JsonUtils.GSON.fromJson(reader, Metadata.class);
                return metadata.getMetadataVersion() != null && metadata.getArtifacts() != null;
            } catch (IOException | JsonParseException e) {
                logger.warn("Invalid metadata.json file", e);
            }
            return false;
        }

        /** {@inheritDoc} */
        @Override
        public Set<String> getSupportedScheme() {
            return Collections.singleton("file");
        }
    }

    private static final class DjlRepositoryFactory implements RepositoryFactory {

        /** {@inheritDoc} */
        @Override
        public Repository newInstance(String name, URI uri) {
            String queryString = uri.getQuery();
            URI djlUri;
            if (queryString != null) {
                djlUri = URI.create("https://mlrepo.djl.ai/?" + queryString);
            } else {
                djlUri = URI.create("https://mlrepo.djl.ai/");
            }

            RemoteRepository repo = new RemoteRepository(name, djlUri);
            String groupId = uri.getHost();
            if (groupId == null) {
                throw new IllegalArgumentException("Invalid djl URL: " + uri);
            }
            String artifactId = parseFilePath(uri);
            if (artifactId.startsWith("/")) {
                artifactId = artifactId.substring(1);
            }
            if (artifactId.isEmpty()) {
                throw new IllegalArgumentException("Invalid djl URL: " + uri);
            }
            String version = null;
            String artifactName = null;
            Matcher m = PATTERN.matcher(artifactId);
            if (m.matches()) {
                artifactId = m.group(1);
                version = m.group(2);
                artifactName = m.group(4);
            }

            ModelZoo zoo = ModelZoo.getModelZoo(groupId);
            if (zoo == null) {
                throw new IllegalArgumentException("ModelZoo not found in classpath: " + groupId);
            }

            ModelLoader loader = zoo.getModelLoader(artifactId);
            if (loader == null) {
                throw new IllegalArgumentException("Invalid djl URL: " + uri);
            }

            MRL mrl =
                    repo.model(loader.getApplication(), groupId, artifactId, version, artifactName);
            repo.addResource(mrl);
            return repo;
        }

        /** {@inheritDoc} */
        @Override
        public Set<String> getSupportedScheme() {
            return Collections.singleton("djl");
        }
    }

    static final class S3RepositoryFactory implements RepositoryFactory {

        /** {@inheritDoc} */
        @Override
        public Repository newInstance(String name, URI uri) {
            try {
                Path path = mount(uri);
                return new SimpleRepository(name, uri, path);
            } catch (IOException | InterruptedException e) {
                throw new IllegalArgumentException("Failed to mount s3 bucket", e);
            }
        }

        /** {@inheritDoc} */
        @Override
        public Set<String> getSupportedScheme() {
            return Collections.singleton("s3");
        }

        static String findS3Fuse() {
            if (System.getProperty("os.name").startsWith("Win")) {
                logger.debug("mount-s3 is not supported on Windows");
                return null;
            }
            String gcsFuse = Utils.getEnvOrSystemProperty("MOUNT_S3", "/usr/bin/mount-s3");
            if (Files.isRegularFile(Paths.get(gcsFuse))) {
                return gcsFuse;
            }
            String path = System.getenv("PATH");
            String[] directories = path.split(File.pathSeparator);
            for (String dir : directories) {
                Path file = Paths.get(dir, "mount-s3");
                if (Files.isRegularFile(file)) {
                    return file.toAbsolutePath().toString();
                }
            }
            return null;
        }

        private static Path mount(URI uri) throws IOException, InterruptedException {
            String bucket = uri.getHost();
            String prefix = uri.getPath();
            if (!prefix.isEmpty()) {
                prefix = prefix.substring(1);
            }
            Path dir = Utils.getCacheDir().toAbsolutePath().normalize();
            dir = dir.resolve("s3").resolve(Utils.hash(uri.toString()));
            String path = dir.toString();
            if (Files.isDirectory(dir)) {
                if (isMounted(path)) {
                    return dir.resolve(prefix);
                }
            } else {
                Files.createDirectories(dir);
            }

            exec(findS3Fuse(), bucket, path);
            return dir.resolve(prefix);
        }
    }

    static final class GcsRepositoryFactory implements RepositoryFactory {

        /** {@inheritDoc} */
        @Override
        public Repository newInstance(String name, URI uri) {
            try {
                Path path = mount(uri);
                return new SimpleRepository(name, uri, path);
            } catch (IOException | InterruptedException e) {
                throw new IllegalArgumentException("Failed to mount gs bucket", e);
            }
        }

        /** {@inheritDoc} */
        @Override
        public Set<String> getSupportedScheme() {
            return Collections.singleton("gs");
        }

        static String findGcsFuse() {
            if (System.getProperty("os.name").startsWith("Win")) {
                logger.debug("gcsfuse is not supported on Windows");
                return null;
            }
            String gcsFuse = Utils.getEnvOrSystemProperty("GCSFUSE", "/usr/bin/gcsfuse");
            if (Files.isRegularFile(Paths.get(gcsFuse))) {
                return gcsFuse;
            }
            String path = System.getenv("PATH");
            String[] directories = path.split(File.pathSeparator);
            for (String dir : directories) {
                Path file = Paths.get(dir, "gcsfuse");
                if (Files.isRegularFile(file)) {
                    return file.toAbsolutePath().toString();
                }
            }
            return null;
        }

        private static Path mount(URI uri) throws IOException, InterruptedException {
            String bucket = uri.getHost();
            String prefix = uri.getPath();
            if (!prefix.isEmpty()) {
                prefix = prefix.substring(1);
            }
            Path dir = Utils.getCacheDir().toAbsolutePath().normalize();
            dir = dir.resolve("gs").resolve(Utils.hash(uri.toString()));
            String path = dir.toString();
            if (Files.isDirectory(dir)) {
                if (isMounted(path)) {
                    return dir.resolve(prefix);
                }
            } else {
                Files.createDirectories(dir);
            }

            exec(findGcsFuse(), "--implicit-dirs", bucket, path);
            return dir.resolve(prefix);
        }
    }
}
