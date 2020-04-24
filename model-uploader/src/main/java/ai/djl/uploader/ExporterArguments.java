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
import com.google.gson.reflect.TypeToken;
import java.io.IOException;
import java.lang.reflect.Type;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.Map;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionGroup;
import org.apache.commons.cli.Options;

public class ExporterArguments {

    private String category;
    private String baseDir;
    private Application application;
    private String name;
    private String groupId;
    private String description;
    private String artifactId;
    private String artifactName;
    private boolean isRemote;
    private Map<String, String> properties;
    private Map<String, Object> arguments;
    private String artifactDir;
    private String pythonPath;
    private String shape;
    private String applicationType;

    public ExporterArguments(CommandLine cmd) {
        if (cmd.hasOption("dataset")) {
            category = "dataset";
        }
        if (cmd.hasOption("gluoncv")) {
            category = "gluoncv";
        }
        if (cmd.hasOption("keras")) {
            category = "keras";
        }
        if (cmd.hasOption("huggingface")) {
            category = "huggingface";
        }
        if (cmd.hasOption("model")) {
            category = "model";
        }
        baseDir = cmd.getOptionValue("output-directory", System.getProperty("user.dir"));
        description = cmd.getOptionValue("description");
        groupId =
                cmd.getOptionValue(
                        "groupid",
                        "dataset".equals(category) ? "ai.djl.basicdataset" : "ai.djl.model");
        artifactId = cmd.getOptionValue("artifact-id");
        artifactName = cmd.getOptionValue("artifact-name", artifactId);
        name = cmd.getOptionValue("name", artifactName);

        String applicationName = cmd.getOptionValue("application", Application.UNDEFINED.getPath());
        if (Application.CV.IMAGE_CLASSIFICATION.getPath().equalsIgnoreCase(applicationName)) {
            application = Application.CV.IMAGE_CLASSIFICATION;
            if (description == null) {
                description = name + " image classification model";
            }
        } else if (Application.CV.OBJECT_DETECTION.getPath().equalsIgnoreCase(applicationName)) {
            application = Application.CV.OBJECT_DETECTION;
            if (description == null) {
                description = name + " object detection model";
            }
        } else if (Application.UNDEFINED.getPath().equalsIgnoreCase(applicationName)) {
            application = Application.UNDEFINED;
            if (description == null) {
                description = name + " model";
            }
        } else {
            throw new IllegalArgumentException("Unsupported application name " + applicationName);
        }
        properties = valueParser(cmd.getOptionValue("property"));
        arguments = parseArguments(cmd.getOptionValue("argument"));
        // TODO: this is not always true, totally relying on model config
        if (arguments.isEmpty() && "imagenet".equals(properties.get("dataset"))) {
            arguments.put("width", 224);
            arguments.put("height", 224);
        }
        artifactDir = cmd.getOptionValue("input-directory");
        isRemote = cmd.hasOption("remote");
        pythonPath = cmd.getOptionValue("python-path");
        shape = cmd.getOptionValue("shape");
        applicationType = cmd.getOptionValue("application-type");
    }

    public static Options getOptions() {
        Options options = new Options();
        OptionGroup og =
                new OptionGroup()
                        .addOption(new Option("dataset", false, "create dataset metadata"))
                        .addOption(new Option("gluoncv", false, "import gluoncv model"))
                        .addOption(new Option("keras", false, "import keras model"))
                        .addOption(new Option("huggingface", false, "import huggingface model"))
                        .addOption(new Option("model", false, "import general model"));
        og.setRequired(true);
        options.addOptionGroup(og);
        options.addRequiredOption(
                "ai",
                "artifact-id",
                true,
                "the way to categorize the artifact, for example, resnet50 has resnet as artifact id");
        options.addOption(
                "an",
                "artifact-name",
                true,
                "the name of the artifact or the name of the model to import");
        options.addOption("o", "output-directory", true, "the output directory");
        options.addOption("n", "name", true, "the name of the metadata");
        options.addOption("g", "groupid", true, "the groupId of the metadata");
        options.addOption("d", "description", true, "the description of the metadata");
        options.addOption("b", "input-directory", true, "the input directory");
        options.addOption(
                "ap",
                "application",
                true,
                "name of the application, example: image classification");
        options.addOption(
                "p",
                "property",
                true,
                "add properties to the metadata, example: --property layers:50,dataset:imagenet");
        options.addOption(
                "a",
                "argument",
                true,
                "add argument to the metadata, example: --argument height:224,width:224");
        options.addOption("r", "remote", false, "set to sync with remote repository");
        options.addOption(
                "py", "python-path", true, "the python path you would like to use for import");
        options.addOption(
                "s",
                "shape",
                true,
                "the input shape of the model, example: (1,3,224,224) or [(1, 3), [(2, 4)]");
        options.addOption(
                "at",
                "application-type",
                true,
                "The application type of the model, example: 'BertForQuestionAnswering' in huggingface");
        return options;
    }

    public String getCategory() {
        return category;
    }

    public String getBaseDir() {
        return baseDir;
    }

    public Application getApplication() {
        return application;
    }

    public String getName() {
        return name;
    }

    public String getGroupId() {
        return groupId;
    }

    public String getDescription() {
        return description;
    }

    public String getArtifactId() {
        return artifactId;
    }

    public String getArtifactName() {
        return artifactName;
    }

    public Map<String, String> getProperties() {
        return properties;
    }

    public Map<String, Object> getArguments() {
        return arguments;
    }

    public Path getArtifactDir() throws IOException {
        if (artifactDir == null) {
            throw new IOException("Please specify --input-directory");
        }
        return Paths.get(artifactDir);
    }

    public String getShape() throws IOException {
        if (shape == null) {
            throw new IOException("Please specify --shape");
        }
        return shape;
    }

    public String getApplicationType() throws IOException {
        if (applicationType == null) {
            throw new IOException("Please specify --application-type");
        }
        return applicationType;
    }

    public String getPythonPath() {
        return pythonPath;
    }

    public boolean isRemote() {
        return isRemote;
    }

    private Map<String, String> valueParser(String value) {
        Map<String, String> result = new LinkedHashMap<>(); // NOPMD
        if (value == null) {
            return result;
        }
        String[] keyValues = value.split(",");
        for (String keyValue : keyValues) {
            String[] keyValueArray = keyValue.split(":");
            result.put(keyValueArray[0], keyValueArray[1]);
        }
        return result;
    }

    private Map<String, Object> parseArguments(String value) {
        Map<String, Object> result = new LinkedHashMap<>(); // NOPMD
        if (value == null || value.isEmpty()) {
            return result;
        }
        Type type = new TypeToken<Map<String, Object>>() {}.getType();
        return MetadataBuilder.GSON.fromJson(value, type);
    }
}
