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

package ai.djl.uploader;

import ai.djl.Application;
import ai.djl.repository.Artifact;
import ai.djl.repository.License;
import ai.djl.repository.Metadata;
import ai.djl.util.Hex;
import ai.djl.util.ZipUtils;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Reader;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.zip.GZIPOutputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** The {@code MetadataBuilder} is designed to help build up metadata for model or dataset. */
public final class MetadataBuilder {

    @SuppressWarnings("rawtypes")
    public static final Gson GSON =
            new GsonBuilder()
                    .setPrettyPrinting()
                    .registerTypeAdapter(
                            LinkedHashMap.class,
                            (JsonDeserializer)
                                    (json, typeOfT, context) -> {
                                        // avoid Gson converting integer type to double
                                        LinkedHashMap<String, Object> m = new LinkedHashMap<>();
                                        JsonObject jo = json.getAsJsonObject();
                                        for (Map.Entry<String, JsonElement> mx : jo.entrySet()) {
                                            m.put(mx.getKey(), mx.getValue());
                                        }
                                        return m;
                                    })
                    .create();

    private static final Logger logger = LoggerFactory.getLogger(MetadataBuilder.class);
    private static final String METADATA_VERSION = "0.1";

    private String artifactVersion = "0.0.1";
    private License license = License.apache();
    private Boolean isSnapshot;
    private Boolean isDataSet = false;
    private String baseDir;
    private Application application;
    private String name;
    private String groupId;
    private String description;
    private String artifactId;
    private String artifactName;
    private Path artifactDir;
    private LinkedHashMap<String, String> properties;
    private LinkedHashMap<String, Object> arguments;

    private MetadataBuilder() {}

    public static MetadataBuilder builder() {
        return new MetadataBuilder();
    }

    /**
     * Sets the group id for this metadata.
     *
     * <p>examples: ai.djl.mxnet, ai.djl.basicdataset
     *
     * @param groupId the group id for this metadata
     * @return builder
     */
    public MetadataBuilder setGroupId(String groupId) {
        this.groupId = groupId;
        return this;
    }

    /**
     * Sets the name for this metadata.
     *
     * <p>examples: Image Classification
     *
     * @param name the title of this metadata
     * @return builder
     */
    public MetadataBuilder setName(String name) {
        this.name = name;
        return this;
    }

    /**
     * Sets the description of this metadata.
     *
     * <p>Please use some words explain what files include in this metadata and how to use them
     *
     * @param description the description of this metadata
     * @return builder
     */
    public MetadataBuilder setDescription(String description) {
        this.description = description;
        return this;
    }

    /**
     * Sets the {@link Application} for this metadata.
     *
     * <p>example: Application.CV.IMAGE_CLASSIFICATION for image classification task
     *
     * @param application model's application
     * @return this builder
     */
    public MetadataBuilder setApplication(Application application) {
        this.application = application;
        return this;
    }

    /**
     * Sets the artifact id for this metadata.
     *
     * <p>this field used to categorized similar models, such as resnet50 and resnet101 all have
     * resnet as the artifact id
     *
     * @param artifactId the artifact id for this metadata
     * @return builder
     */
    public MetadataBuilder setArtifactId(String artifactId) {
        this.artifactId = artifactId;
        return this;
    }

    /**
     * Sets the name of the artifact.
     *
     * <p>this name will replace the name of the model
     *
     * @param artifactName the name of the artifact
     * @return builder
     */
    public MetadataBuilder setArtifactName(String artifactName) {
        this.artifactName = artifactName;
        return this;
    }

    /**
     * Sets the directory of where artifacts stored.
     *
     * @param artifactDir full path to the directory
     * @return builder
     */
    public MetadataBuilder setArtifactDir(Path artifactDir) {
        if (Files.isDirectory(artifactDir)) {
            this.artifactDir = artifactDir;
        } else {
            throw new IllegalArgumentException("File path is not a directory " + artifactDir);
        }
        return this;
    }

    /**
     * Sets the directory where the output repository stored.
     *
     * @param baseDir full path to the output repository
     * @return builder
     */
    public MetadataBuilder setBaseDir(String baseDir) {
        this.baseDir = baseDir;
        return this;
    }

    /**
     * Sets the artifact version.
     *
     * <p>By default, we will try to assign one for you and bump up the version if found some
     * artifacts contains same name
     *
     * @param artifactVersion the artifact version
     * @return builder
     */
    public MetadataBuilder optArtifactVersion(String artifactVersion) {
        this.artifactVersion = artifactVersion;
        return this;
    }

    /**
     * Labels this artifact as snapshot.
     *
     * @return builder
     */
    public MetadataBuilder optIsSnapshot() {
        this.isSnapshot = true;
        return this;
    }

    /**
     * Assign {@link License} to this metadata.
     *
     * <p>Apache 2.0 licence will be used by default
     *
     * @param license the licence applied to the metadata
     * @return builder
     */
    public MetadataBuilder optLicense(License license) {
        this.license = license;
        return this;
    }

    /**
     * Labels this artifact as dataset.
     *
     * @return builder
     */
    public MetadataBuilder optIsDataset() {
        this.isDataSet = true;
        return this;
    }

    /**
     * Adds a property to the artifact.
     *
     * <p>Properties are used by {@link ai.djl.repository.zoo.Criteria} to classify and find the
     * right model
     *
     * @param key the key of the property
     * @param value the value of the property
     * @return builder
     */
    public MetadataBuilder addProperty(String key, String value) {
        if (properties == null) {
            properties = new LinkedHashMap<>();
        }
        properties.put(key, value);
        return this;
    }

    /**
     * Adds properties to the artifact.
     *
     * @param properties properties to classify the example
     * @return builder
     */
    public MetadataBuilder addProperties(Map<String, String> properties) {
        if (this.properties == null) {
            this.properties = new LinkedHashMap<>();
        }
        this.properties.putAll(properties);
        return this;
    }

    /**
     * Adds an argument to the artifact.
     *
     * <p>Arguments are helpful for people to get the information of the model or dataset. Such as
     * input shape, text size, etc
     *
     * @param key the name of the argument
     * @param value the object (needs to be serialized) assigned to this argument
     * @return builder
     */
    public MetadataBuilder addArgument(String key, Object value) {
        if (arguments == null) {
            arguments = new LinkedHashMap<>();
        }
        arguments.put(key, value);
        return this;
    }

    /**
     * Adds arguments to the artifact.
     *
     * @param arguments arguments to use the model
     * @return builder
     */
    public MetadataBuilder addArguments(Map<String, Object> arguments) {
        if (this.arguments == null) {
            this.arguments = new LinkedHashMap<>();
        }
        this.arguments.putAll(arguments);
        return this;
    }

    /**
     * Builds the metadata locally.
     *
     * <p>This function will create or update the metadata, compress all artifacts and copy them
     * into the formatted location in the repository.
     *
     * @return Metadata
     * @throws IOException failed to create or copy files
     */
    public Metadata buildLocal() throws IOException {
        String metadataFileName = "metadata.json";
        Path targetDir =
                Paths.get(
                        baseDir,
                        "mlrepo",
                        isDataSet ? "dataset" : "model",
                        application.getPath(),
                        groupId.replace('.', '/'),
                        artifactId);
        Path artifactDir = targetDir.resolve(artifactName);
        Files.createDirectories(artifactDir);
        Artifact artifact = constructArtifact(artifactDir);
        Path metadataPath = targetDir.resolve(metadataFileName);
        Metadata metadata;
        if (Files.exists(metadataPath)) {
            logger.info("Found exsisting metadata, try to add new artifact");
            metadata = extendExisting(metadataPath, artifact);
        } else {
            logger.info("No existing metadata found, try to create a new one");
            metadata = writeNew(artifact);
        }
        Files.deleteIfExists(metadataPath);
        try (BufferedWriter myWriter = Files.newBufferedWriter(metadataPath)) {
            myWriter.write(GSON.toJson(metadata, Metadata.class));
        }
        return metadata;
    }

    /**
     * Fetch the existing metadata in model zoo and update them locally.
     *
     * <p>If the metadata is not found in model zoo, a metadata will be created
     *
     * @return Metadata
     * @throws IOException failed to create or copy files
     */
    public Metadata buildExternal() throws IOException {
        String fileName = "metadata.json";
        String groupPath = groupId.replace('.', '/');
        String reconstructedURI =
                "https://mlrepo.djl.ai/"
                        + (isDataSet ? "dataset/" : "model/")
                        + application.getPath()
                        + '/'
                        + groupPath
                        + '/'
                        + artifactId;
        Path targetDir =
                Paths.get(
                        baseDir,
                        "mlrepo",
                        isDataSet ? "dataset" : "model",
                        application.getPath(),
                        groupPath,
                        artifactId);
        Path metadata = targetDir.resolve(fileName);
        if (Files.exists(metadata)) {
            throw new IllegalStateException(
                    "metadata found in the local repository, please remove: " + metadata);
        }
        try {
            URL metadataURL = new URL(reconstructedURI + '/' + fileName);
            HttpURLConnection huc = (HttpURLConnection) metadataURL.openConnection();
            huc.setRequestMethod("HEAD");
            int responseCode = huc.getResponseCode();
            if (HttpURLConnection.HTTP_OK == responseCode) {
                Files.createDirectories(targetDir);
                try (InputStream is = metadataURL.openStream()) {
                    Files.copy(is, metadata, StandardCopyOption.REPLACE_EXISTING);
                }
            }
            return buildLocal();
        } catch (MalformedURLException e) {
            throw new IllegalArgumentException("The properties to create URL is invalid", e);
        }
    }

    private Metadata extendExisting(Path metadataPath, Artifact artifact) throws IOException {
        try (Reader reader = Files.newBufferedReader(metadataPath)) {
            Metadata metadata = GSON.fromJson(reader, Metadata.class);
            metadata.addArtifact(artifact);
            return metadata;
        }
    }

    private Metadata writeNew(Artifact artifact) {
        Metadata metadata = new Metadata();
        metadata.setArtifactId(artifactId);
        metadata.setDescription(description);
        metadata.setGroupId(groupId);
        metadata.setName(name);
        metadata.setApplication(application);
        metadata.setMetadataVersion(METADATA_VERSION);
        String[] tokens = groupId.split("\\.");
        String engine = tokens[tokens.length - 1];
        if (isDataSet) {
            metadata.setWebsite("http://www.djl.ai/" + engine);
        } else {
            metadata.setWebsite("http://www.djl.ai/" + engine + "/model-zoo");
        }
        metadata.addLicense(license);
        metadata.addArtifact(artifact);
        return metadata;
    }

    private Artifact constructArtifact(Path destination) throws IOException {
        Artifact artifact = new Artifact();
        artifact.setArguments(arguments);
        artifact.setName(artifactName);
        artifact.setProperties(properties);
        if (isSnapshot != null) {
            artifact.setSnapshot(isSnapshot);
        }
        artifact.setFiles(constructFiles(destination));
        Map<String, Artifact.Item> fileList = artifact.getFiles();
        if (!isDataSet && !fileList.containsKey("model") && !fileList.containsKey("symbol")) {
            throw new IllegalStateException("Model not found in files! " + fileList.keySet());
        }
        artifact.setVersion(artifactVersion);
        return artifact;
    }

    private Map<String, Artifact.Item> constructFiles(Path destination) throws IOException {
        File[] listFiles = artifactDir.toFile().listFiles();
        if (listFiles == null) {
            throw new FileNotFoundException("File not found in dir: " + artifactDir);
        }
        while (existCollideFile(listFiles, destination)) {
            String[] digits = artifactVersion.split("\\.");
            // TODO: check digit go to 10
            digits[digits.length - 1] =
                    String.valueOf(Integer.parseInt(digits[digits.length - 1]) + 1);
            artifactVersion = String.join(".", digits);
            logger.info("Found existing file(s), bump up version to " + artifactVersion);
        }
        Files.createDirectories(destination.resolve(artifactVersion));
        Map<String, Artifact.Item> files = new ConcurrentHashMap<>();
        for (File file : listFiles) {
            if (file.isHidden()) {
                continue;
            }
            String[] names = nameAnalyzer(file.getName());
            String uri;
            if (file.isDirectory()) {
                uri = artifactVersion + '/' + names[1] + ".zip";
                ZipUtils.zip(file.toPath(), destination.resolve(uri));
            } else if (isCompressed(file.getName())) {
                uri = artifactVersion + '/' + names[1];
                Files.copy(file.toPath(), destination.resolve(uri));
            } else {
                uri = artifactVersion + '/' + names[1] + ".gz";
                gzipFile(file, destination.resolve(uri));
            }
            File copiedFile = destination.resolve(uri).toFile();
            Artifact.Item item = new Artifact.Item();
            item.setSha1Hash(getSha1Sum(copiedFile));
            item.setSize(copiedFile.length());
            item.setUri(artifactName + '/' + uri);
            files.put(names[0], item);
        }
        return files;
    }

    private void gzipFile(File file, Path destination) throws IOException {
        InputStream is = Files.newInputStream(file.toPath());

        OutputStream os = Files.newOutputStream(destination);
        GZIPOutputStream gzipOS = new GZIPOutputStream(os);
        byte[] buffer = new byte[40960];
        int len;
        while ((len = is.read(buffer)) != -1) {
            gzipOS.write(buffer, 0, len);
        }
        // close resources
        gzipOS.close();
        os.close();
        is.close();
    }

    private boolean existCollideFile(File[] listFiles, Path targetDir) {
        for (File file : listFiles) {
            if (file.isHidden()) {
                continue;
            }
            String[] names = nameAnalyzer(file.getName());
            String uri;
            if (file.isDirectory()) {
                uri = artifactVersion + '/' + names[1] + ".zip";
            } else if (isCompressed(file.getName())) {
                uri = artifactVersion + '/' + names[1];
            } else {
                uri = artifactVersion + '/' + names[1] + ".gz";
            }
            if (Files.exists(targetDir.resolve(uri))) {
                logger.debug("Found existing file: " + uri);
                return true;
            }
        }
        return false;
    }

    private String getSha1Sum(File file) throws IOException {
        MessageDigest md;
        try {
            md = MessageDigest.getInstance("SHA1");
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError("SHA1 algorithm not found.", e);
        }
        try (DigestInputStream dis =
                new DigestInputStream(Files.newInputStream(file.toPath()), md)) {
            byte[] buf = new byte[40960];
            int data = dis.read(buf);
            while (data != -1) {
                data = dis.read(buf);
            }
            return Hex.toHexString(dis.getMessageDigest().digest());
        }
    }

    private String[] nameAnalyzer(String fileName) {
        String fileClassifier;
        String realName;
        if (isDataSet) {
            realName = fileName;
            fileClassifier = fileName.split("\\.")[0];
            return new String[] {fileClassifier, realName};
        }
        if (fileName.endsWith("-symbol.json")) {
            // mxnet
            realName = artifactName + "-symbol.json";
            fileClassifier = "symbol";
        } else if (fileName.endsWith(".params")) {
            // mxnet
            String[] names = fileName.split("-");
            String epoch = names[names.length - 1];
            realName = artifactName + "-" + epoch;
            fileClassifier = "parameters";
        } else if (fileName.endsWith(".pt")) {
            // pytorch
            realName = artifactName + ".pt";
            fileClassifier = "model";
        } else if (fileName.endsWith(".zip")) {
            // tensorflow
            realName = artifactName + ".zip";
            fileClassifier = "model";
        } else {
            realName = fileName;
            fileClassifier = fileName.split("\\.")[0];
        }
        return new String[] {fileClassifier, realName};
    }

    private boolean isCompressed(String name) {
        String[] compressedExtensions = {".gz", ".z", ".tar", ".tgz", ".zip", ".7z"};
        for (String extension : compressedExtensions) {
            if (name.endsWith(extension)) {
                return true;
            }
        }
        return false;
    }
}
