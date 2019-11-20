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
package ai.djl.examples.inference.util;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;

/** A class represents parsed command line arguments. */
public class Arguments {

    private String modelDir;
    private String modelName;
    private String imageFile;
    private String outputDir;
    private Map<String, String> criteria;
    private int duration;
    private int iteration = 1;
    private int threads;

    private boolean isImperative;

    public Arguments(CommandLine cmd) {
        modelDir = cmd.getOptionValue("model-dir");
        modelName = cmd.getOptionValue("model-name");
        outputDir = cmd.getOptionValue("output-dir");
        imageFile = cmd.getOptionValue("image");
        if (cmd.hasOption("duration")) {
            duration = Integer.parseInt(cmd.getOptionValue("duration"));
        }
        if (cmd.hasOption("iteration")) {
            iteration = Integer.parseInt(cmd.getOptionValue("iteration"));
        }
        if (cmd.hasOption("threads")) {
            threads = Integer.parseInt(cmd.getOptionValue("threads"));
        } else {
            threads = Runtime.getRuntime().availableProcessors() * 2 - 1;
        }
        if (cmd.hasOption("criteria")) {
            Type type = new TypeToken<Map<String, String>>() {}.getType();
            criteria = new Gson().fromJson(cmd.getOptionValue("criteria"), type);
        }
        isImperative = cmd.hasOption("imperative");
    }

    public static Options getOptions() {
        Options options = new Options();
        options.addOption(
                Option.builder("p")
                        .longOpt("model-dir")
                        .hasArg()
                        .argName("MODEL-DIR")
                        .desc("Path to the model directory.")
                        .build());
        options.addOption(
                Option.builder("n")
                        .longOpt("model-name")
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
                        .desc("Duration of the test in minutes.")
                        .build());
        options.addOption(
                Option.builder("c")
                        .longOpt("iteration")
                        .hasArg()
                        .argName("ITERATION")
                        .desc("Number of iterations in each test.")
                        .build());
        options.addOption(
                Option.builder("t")
                        .longOpt("threads")
                        .hasArg()
                        .argName("NUMBER_THREADS")
                        .desc("Number of inference threads.")
                        .build());
        options.addOption(
                Option.builder("o")
                        .longOpt("output-dir")
                        .hasArg()
                        .argName("OUTPUT-DIR")
                        .desc("Directory for output logs.")
                        .build());
        options.addOption(
                Option.builder("r")
                        .longOpt("criteria")
                        .hasArg()
                        .argName("CRITERIA")
                        .desc("The criteria used for the model.")
                        .build());
        options.addOption(
                Option.builder("m")
                        .longOpt("imperative")
                        .argName("IMPERATIVE")
                        .desc("set to true to use imperative model.")
                        .build());
        return options;
    }

    public int getDuration() {
        return duration;
    }

    public Path getModelDir() throws IOException {
        if (modelDir == null) {
            throw new IOException("Please specify --model-dir");
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
            Path path = Paths.get("src/test/resources/kitten.jpg");
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

    public boolean isImperative() {
        return isImperative;
    }

    public int getIteration() {
        return iteration;
    }

    public int getThreads() {
        return threads;
    }

    public String getOutputDir() {
        return outputDir;
    }

    public Map<String, String> getCriteria() {
        return criteria;
    }
}
