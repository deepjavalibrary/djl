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
package ai.djl.examples.inference.benchmark.util;

import ai.djl.engine.Engine;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.JsonUtils;
import ai.djl.util.PairList;
import com.google.gson.reflect.TypeToken;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;

/** A class represents parsed command line arguments. */
public class Arguments {

    private String artifactId;
    private String modelName;
    private String imageFile;
    private String outputDir;
    private Map<String, String> criteria;
    private int duration;
    private int iteration;
    private int threads;
    private int delay;
    private PairList<DataType, Shape> inputShapes;
    private boolean help;

    public Arguments(CommandLine cmd) {
        artifactId = cmd.getOptionValue("artifact-id");
        modelName = cmd.getOptionValue("model-name");
        outputDir = cmd.getOptionValue("output-dir");
        imageFile = cmd.getOptionValue("image");
        help = cmd.hasOption("help");
        inputShapes = new PairList<>();

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
                threads = Runtime.getRuntime().availableProcessors() * 2 - 1;
            }
        } else {
            threads = Runtime.getRuntime().availableProcessors() * 2 - 1;
        }
        if (cmd.hasOption("criteria")) {
            Type type = new TypeToken<Map<String, String>>() {}.getType();
            criteria = JsonUtils.GSON.fromJson(cmd.getOptionValue("criteria"), type);
        }
        if (cmd.hasOption("delay")) {
            delay = Integer.parseInt(cmd.getOptionValue("delay"));
        }
        if (cmd.hasOption("input-shapes")) {
            String shape = cmd.getOptionValue("input-shapes");
            if (shape.contains("(")) {
                Pattern pattern =
                        Pattern.compile("\\((\\s*(\\d+)([,\\s]+\\d+)*\\s*)\\)([sfduislbS]?)");
                Matcher matcher = pattern.matcher(shape);
                while (matcher.find()) {
                    String[] tokens = matcher.group(1).split(",");
                    long[] array = Arrays.stream(tokens).mapToLong(Long::parseLong).toArray();
                    DataType dataType;
                    String dataTypeStr = matcher.group(4);
                    if (dataTypeStr == null || dataTypeStr.isEmpty()) {
                        dataType = DataType.FLOAT32;
                    } else {
                        switch (dataTypeStr) {
                            case "s":
                                dataType = DataType.FLOAT16;
                                break;
                            case "d":
                                dataType = DataType.FLOAT64;
                                break;
                            case "u":
                                dataType = DataType.UINT8;
                                break;
                            case "b":
                                dataType = DataType.INT8;
                                break;
                            case "i":
                                dataType = DataType.INT32;
                                break;
                            case "l":
                                dataType = DataType.INT64;
                                break;
                            case "B":
                                dataType = DataType.BOOLEAN;
                                break;
                            case "f":
                                dataType = DataType.FLOAT32;
                                break;
                            default:
                                throw new IllegalArgumentException("Invalid input-shape: " + shape);
                        }
                    }
                    inputShapes.add(dataType, new Shape(array));
                }
            } else {
                String[] tokens = shape.split(",");
                long[] shapes = Arrays.stream(tokens).mapToLong(Long::parseLong).toArray();
                inputShapes.add(DataType.FLOAT32, new Shape(shapes));
            }
        }
    }

    public static Options getOptions() {
        Options options = new Options();
        options.addOption(
                Option.builder("h").longOpt("help").hasArg(false).desc("Print this help.").build());
        options.addOption(
                Option.builder("a")
                        .longOpt("artifact-id")
                        .hasArg()
                        .argName("ARTIFACT-ID")
                        .desc("Model artifact id.")
                        .build());
        options.addOption(
                Option.builder("n")
                        .longOpt("model-name")
                        .hasArg()
                        .argName("MODEL-NAME")
                        .desc("Model name.")
                        .build());
        options.addOption(
                Option.builder("s")
                        .longOpt("input-shapes")
                        .hasArg()
                        .argName("INPUT-SHAPES")
                        .desc("Input data shapes for non-CV model.")
                        .build());
        options.addOption(
                Option.builder("i")
                        .longOpt("image")
                        .hasArg()
                        .argName("IMAGE")
                        .desc("Image file path for benchmarking CV model.")
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
                        .desc("Number of total iterations (per thread).")
                        .build());
        options.addOption(
                Option.builder("t")
                        .longOpt("threads")
                        .hasArg()
                        .argName("NUMBER_THREADS")
                        .desc("Number of inference threads.")
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
        options.addOption(
                Option.builder("r")
                        .longOpt("criteria")
                        .hasArg()
                        .argName("CRITERIA")
                        .desc("The criteria (json string) used for searching the model.")
                        .build());
        return options;
    }

    public int getDuration() {
        return duration;
    }

    public String getModelName() {
        return modelName;
    }

    public String getArtifactId() {
        if (System.getProperty("ai.djl.repository.zoo.location") != null) {
            return "ai.djl.localmodelzoo:";
        }

        if (artifactId != null) {
            return artifactId;
        }

        switch (Engine.getInstance().getEngineName()) {
            case "PyTorch":
                return "ai.djl.pytorch:resnet";
            case "TensorFlow":
                return "ai.djl.tensorflow:resnet";
            case "MXNet":
            default:
                return "ai.djl.mxnet:resnet";
        }
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

    public int getIteration() {
        return iteration;
    }

    public int getThreads() {
        return threads;
    }

    public String getOutputDir() {
        if (outputDir == null) {
            outputDir = "build";
        }
        return outputDir;
    }

    public Map<String, String> getCriteria() {
        return criteria;
    }

    public Class<?> getInputClass() {
        if (inputShapes.isEmpty()) {
            return Image.class;
        }
        return NDList.class;
    }

    public Class<?> getOutputClass() {
        if (inputShapes.isEmpty()) {
            if (artifactId != null && artifactId.contains("ssd")) {
                return DetectedObjects.class;
            }
            return Classifications.class;
        }
        return NDList.class;
    }

    public Object getInputData() throws IOException {
        if (inputShapes.isEmpty()) {
            return ImageFactory.getInstance().fromFile(getImageFile());
        }
        return null;
    }

    public int getDelay() {
        return delay;
    }

    public PairList<DataType, Shape> getInputShapes() {
        return inputShapes;
    }

    public boolean hasHelp() {
        return help;
    }
}
