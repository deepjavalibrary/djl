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
package ai.djl.benchmark;

import ai.djl.engine.Engine;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.PairList;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionGroup;
import org.apache.commons.cli.Options;

/** A class represents parsed command line arguments. */
public class Arguments {

    private String modelUrl;
    private String modelName;
    private String engine;
    private String outputDir;
    private int duration;
    private int iteration;
    private int threads;
    private int maxGpus;
    private int delay;
    private PairList<DataType, Shape> inputShapes;

    /**
     * Constructs a {@code Arguments} instance.
     *
     * @param cmd command line options
     */
    Arguments(CommandLine cmd) {
        if (cmd.hasOption("model-path")) {
            String modelPath = cmd.getOptionValue("model-path");
            Path path = Paths.get(modelPath);
            try {
                modelUrl = path.toUri().toURL().toExternalForm();
            } catch (IOException e) {
                throw new IllegalArgumentException("Invalid model-path: " + modelPath, e);
            }
        } else if (cmd.hasOption("model-url")) {
            modelUrl = cmd.getOptionValue("model-url");
        }

        modelName = cmd.getOptionValue("model-name");
        outputDir = cmd.getOptionValue("output-dir");

        if (cmd.hasOption("engine")) {
            engine = cmd.getOptionValue("engine");
        } else {
            engine = Engine.getDefaultEngineName();
        }

        if (cmd.hasOption("duration")) {
            duration = Integer.parseInt(cmd.getOptionValue("duration"));
        }
        iteration = 1;
        if (cmd.hasOption("iteration")) {
            iteration = Integer.parseInt(cmd.getOptionValue("iteration"));
        }
        if (cmd.hasOption("threads")) {
            threads = Integer.parseInt(cmd.getOptionValue("threads"));
            if (threads <= 0) {
                threads = Runtime.getRuntime().availableProcessors();
            }
        }
        if (cmd.hasOption("gpus")) {
            maxGpus = Integer.parseInt(cmd.getOptionValue("gpus"));
            if (maxGpus < 0) {
                maxGpus = Integer.MAX_VALUE;
            }
        } else {
            maxGpus = Integer.MAX_VALUE;
        }
        if (cmd.hasOption("delay")) {
            delay = Integer.parseInt(cmd.getOptionValue("delay"));
        }

        String shape = cmd.getOptionValue("input-shapes");
        inputShapes = NDListGenerator.parseShape(shape);
    }

    static Options getOptions() {
        Options options = new Options();
        options.addOption(
                Option.builder("h").longOpt("help").hasArg(false).desc("Print this help.").build());
        OptionGroup artifactGroup = new OptionGroup();
        artifactGroup.setRequired(true);
        artifactGroup.addOption(
                Option.builder("p")
                        .longOpt("model-path")
                        .hasArg()
                        .argName("MODEL-PATH")
                        .desc("Model directory file path.")
                        .build());
        artifactGroup.addOption(
                Option.builder("u")
                        .longOpt("model-url")
                        .hasArg()
                        .argName("MODEL-URL")
                        .desc("Model archive file URL.")
                        .build());
        options.addOptionGroup(artifactGroup);
        options.addOption(
                Option.builder("n")
                        .longOpt("model-name")
                        .hasArg()
                        .argName("MODEL-NAME")
                        .desc("Specify model file name.")
                        .build());
        options.addOption(
                Option.builder("e")
                        .longOpt("engine")
                        .hasArg()
                        .argName("ENGINE-NAME")
                        .desc("Choose an Engine for the benchmark.")
                        .build());
        options.addOption(
                Option.builder("s")
                        .required()
                        .longOpt("input-shapes")
                        .hasArg()
                        .argName("INPUT-SHAPES")
                        .desc("Input data shapes for the model.")
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
                        .desc("Number of total iterations.")
                        .build());
        options.addOption(
                Option.builder("t")
                        .longOpt("threads")
                        .hasArg()
                        .argName("NUMBER_THREADS")
                        .desc("Number of inference threads.")
                        .build());
        options.addOption(
                Option.builder("g")
                        .longOpt("gpus")
                        .hasArg()
                        .argName("NUMBER_GPUS")
                        .desc("Number of GPUS to run multithreading inference.")
                        .build());
        options.addOption(
                Option.builder("l")
                        .longOpt("delay")
                        .hasArg()
                        .argName("DELAY")
                        .desc("Delay of incremental threads.")
                        .build());
        options.addOption(
                Option.builder("o")
                        .longOpt("output-dir")
                        .hasArg()
                        .argName("OUTPUT-DIR")
                        .desc("Directory for output logs.")
                        .build());
        return options;
    }

    static boolean hasHelp(String[] args) {
        List<String> list = Arrays.asList(args);
        return list.contains("-h") || list.contains("help");
    }

    static void printHelp(String msg, Options options) {
        HelpFormatter formatter = new HelpFormatter();
        formatter.setSyntaxPrefix("");
        formatter.setLeftPadding(1);
        formatter.setWidth(120);
        formatter.printHelp(msg, options);
    }

    int getDuration() {
        return duration;
    }

    String getEngine() {
        return engine;
    }

    String getModelUrl() {
        return modelUrl;
    }

    String getModelName() {
        return modelName;
    }

    int getIteration() {
        return iteration;
    }

    int getThreads() {
        return threads;
    }

    int getMaxGpus() {
        return maxGpus;
    }

    String getOutputDir() {
        if (outputDir == null) {
            outputDir = "build";
        }
        return outputDir;
    }

    int getDelay() {
        return delay;
    }

    PairList<DataType, Shape> getInputShapes() {
        return inputShapes;
    }
}
