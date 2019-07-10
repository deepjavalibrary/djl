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

package software.amazon.ai.example.util;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionGroup;
import org.apache.commons.cli.Options;

/** A class represents parsed command line arguments. */
public class Arguments {

    private String modelDir;
    private String modelUrl;
    private String modelName;
    private String imageFile;
    private String logDir;
    private int duration;
    private int iteration = 1;

    public Arguments(CommandLine cmd) {
        modelDir = cmd.getOptionValue("model-dir");
        modelUrl = cmd.getOptionValue("model-url");
        modelName = cmd.getOptionValue("model-name");
        logDir = cmd.getOptionValue("log-dir");
        imageFile = cmd.getOptionValue("image");
        if (cmd.hasOption("duration")) {
            duration = Integer.parseInt(cmd.getOptionValue("duration"));
        }
        if (cmd.hasOption("iteration")) {
            iteration = Integer.parseInt(cmd.getOptionValue("iteration"));
        }
    }

    public static Options getOptions() {
        Options options = new Options();
        OptionGroup group = new OptionGroup();
        group.addOption(
                        Option.builder("u")
                                .longOpt("model-url")
                                .hasArg()
                                .argName("MODEL-URL")
                                .desc("URL to download model archive.")
                                .build())
                .addOption(
                        Option.builder("p")
                                .longOpt("model-dir")
                                .hasArg()
                                .argName("MODEL-DIR")
                                .desc("Path to the model directory.")
                                .build());
        options.addOptionGroup(group);
        options.addOption(
                Option.builder("n")
                        .longOpt("model-name")
                        .required()
                        .hasArg()
                        .argName("MODEL-NAME")
                        .desc("Model name.")
                        .build());
        options.addOption(
                Option.builder("i")
                        .longOpt("image")
                        .hasArg()
                        .argName("IMAGE")
                        .desc("Image file.")
                        .build());
        options.addOption(
                Option.builder("d")
                        .longOpt("duration")
                        .hasArg()
                        .argName("DURATION")
                        .desc("Duration of the test.")
                        .build());
        options.addOption(
                Option.builder("c")
                        .longOpt("iteration")
                        .hasArg()
                        .argName("ITERATION")
                        .desc("Number of iterations in each test.")
                        .build());
        options.addOption(
                Option.builder("l")
                        .longOpt("log-dir")
                        .hasArg()
                        .argName("LOG-DIR")
                        .desc("Directory for output logs.")
                        .build());
        return options;
    }

    public Path getModelDir() throws IOException {
        if (modelDir == null) {
            ModelInfo modelInfo;
            if (modelUrl == null) {
                modelInfo = ModelInfo.getModel(modelName);
                if (modelInfo == null) {
                    throw new IOException("Please specify --model-dir or --model-url");
                }
            } else {
                modelInfo = new ModelInfo(modelName, modelUrl);
            }
            modelInfo.download();
            return modelInfo.getDownloadDir();
        }

        Path path = Paths.get(modelDir);
        if (Files.notExists(path)) {
            throw new FileNotFoundException("model directory not found: " + modelDir);
        }
        return path;
    }

    public String getModelName() {
        return modelName;
    }

    public Path getImageFile() throws FileNotFoundException {
        if (imageFile == null) {
            Path path = Paths.get(modelDir, "kitten.jpg");
            if (Files.notExists(path)) {
                throw new FileNotFoundException("Missing --image parameter.");
            }
            return path;
        }
        Path path = Paths.get(imageFile);
        if (Files.notExists(path)) {
            throw new FileNotFoundException("image file not found: " + imageFile);
        }
        return path;
    }

    public int getDuration() {
        return duration;
    }

    public int getIteration() {
        return iteration;
    }

    public String getLogDir() {
        return logDir;
    }
}
