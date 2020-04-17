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
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
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
    private Map<String, String> property;
    private Map<String, String> argument;
    private String artifactDir;
    private String pythonPath;
    private String shape;

    public ExporterArguments(CommandLine cmd) {
        if (cmd.hasOption("dataset")) {
            category = "dataset";
        }
        if (cmd.hasOption("gluoncv")) {
            category = "gluoncv";
        }
        if (cmd.hasOption("model")) {
            category = "model";
        }
        baseDir = cmd.getOptionValue("output-directory", System.getProperty("user.dir"));
        String applicationName = cmd.getOptionValue("application");
        if (applicationName == null) {
            application = Application.UNDEFINED;
        } else {
            switch (applicationName) {
                case "image classification":
                    application = Application.CV.IMAGE_CLASSIFICATION;
                    break;
                case "object detection":
                    application = Application.CV.OBJECT_DETECTION;
                    break;
                default:
                    throw new IllegalArgumentException(
                            "Unsupported application name " + applicationName);
            }
        }
        description = cmd.getOptionValue("description", "No description provided");
        name = cmd.getOptionValue("name", "No Name provided");
        groupId =
                cmd.getOptionValue(
                        "groupid",
                        "dataset".equals(category) ? "ai.djl.basicdataset" : "ai.djl.model");
        artifactName = cmd.getOptionValue("artifact-name");
        artifactId = cmd.getOptionValue("artifact-id");
        property = valueParser(cmd.getOptionValue("property"));
        argument = valueParser(cmd.getOptionValue("argument"));
        artifactDir = cmd.getOptionValue("input-directory");
        isRemote = cmd.hasOption("remote");
        pythonPath = cmd.getOptionValue("python-path");
        shape = cmd.getOptionValue("shape");
    }

    public static Options getOptions() {
        Options options = new Options();
        OptionGroup og =
                new OptionGroup()
                        .addOption(new Option("dataset", false, "create dataset metadata"))
                        .addOption(new Option("gluoncv", false, "import gluoncv model"))
                        .addOption(new Option("model", false, "import general model"));
        og.setRequired(true);
        options.addOptionGroup(og);
        options.addRequiredOption(
                "an",
                "artifact-name",
                true,
                "the name of the artifact or the name of the model to import");
        options.addRequiredOption(
                "ai",
                "artifact-id",
                true,
                "the way to categorize the artifact, for example, resnet50 has resnet as artifact id");
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
                "s", "shape", true, "the input shape of the model, example: (1,3,224,224)");
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

    public Map<String, String> getProperty() {
        return property;
    }

    public Map<String, String> getArgument() {
        return argument;
    }

    public String getArtifactDir() throws IOException {
        if (artifactDir == null) {
            throw new IOException("Please specify --input-directory");
        }
        return artifactDir;
    }

    public String getShape() throws IOException {
        if (shape == null) {
            throw new IOException("Please specify --shape");
        }
        return shape;
    }

    public String getPythonPath() {
        return pythonPath;
    }

    public boolean isRemote() {
        return isRemote;
    }

    private Map<String, String> valueParser(String value) {
        Map<String, String> result = new ConcurrentHashMap<>();
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
}
