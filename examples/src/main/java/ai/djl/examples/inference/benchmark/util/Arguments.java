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
import ai.djl.ndarray.types.Shape;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Map;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionGroup;
import org.apache.commons.cli.Options;

/** A class represents parsed command line arguments. */
public class Arguments {

    private String modelDir;
    private String artifactId;
    private String imageFile;
    private String outputDir;
    private Map<String, String> criteria;
    private int duration;
    private int iteration;
    private int threads;
    private String inputClass;
    private String outputClass;
    private Shape inputShape;

    public Arguments(CommandLine cmd) {
        modelDir = cmd.getOptionValue("model-dir");
        artifactId = cmd.getOptionValue("artifact-id");
        outputDir = cmd.getOptionValue("output-dir");
        imageFile = cmd.getOptionValue("image");
        inputClass = cmd.getOptionValue("input-class");
        outputClass = cmd.getOptionValue("output-class");
        if (cmd.hasOption("duration")) {
            duration = Integer.parseInt(cmd.getOptionValue("duration"));
        }
        iteration = 1;
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
        if (cmd.hasOption("input-shape")) {
            String shape = cmd.getOptionValue("input-shape");
            String[] tokens = shape.split(",");
            long[] shapes = Arrays.stream(tokens).mapToLong(Long::parseLong).toArray();
            inputShape = new Shape(shapes);
        }
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
                        .longOpt("artifact-id")
                        .hasArg()
                        .argName("ARTIFACT-ID")
                        .desc("Model artifact id.")
                        .build());
        options.addOption(
                Option.builder("ic")
                        .longOpt("input-class")
                        .hasArg()
                        .argName("INPUT-CLASS")
                        .desc("Input class type.")
                        .build());
        options.addOption(
                Option.builder("is")
                        .longOpt("input-shape")
                        .hasArg()
                        .argName("INPUT-SHAPE")
                        .desc("Input data shape.")
                        .build());
        options.addOption(
                Option.builder("oc")
                        .longOpt("output-class")
                        .hasArg()
                        .argName("OUTPUT-CLASS")
                        .desc("Output class type.")
                        .build());
        options.addOption(
                Option.builder("i")
                        .longOpt("image")
                        .hasArg()
                        .argName("IMAGE")
                        .desc("Image file.")
                        .build());
        options.addOptionGroup(
                new OptionGroup()
                        .addOption(
                                Option.builder("d")
                                        .longOpt("duration")
                                        .hasArg()
                                        .argName("DURATION")
                                        .desc("Duration of the test in minutes.")
                                        .build())
                        .addOption(
                                Option.builder("c")
                                        .longOpt("iteration")
                                        .hasArg()
                                        .argName("ITERATION")
                                        .desc("Number of total iterations.")
                                        .build()));
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

    public String getArtifactId() {
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
        return outputDir;
    }

    public Map<String, String> getCriteria() {
        return criteria;
    }

    public Class<?> getInputClass() throws ClassNotFoundException {
        if (inputClass == null) {
            return Image.class;
        }
        return Class.forName(inputClass);
    }

    public Class<?> getOutputClass() throws ClassNotFoundException {
        if (outputClass == null) {
            if (artifactId != null && artifactId.contains("ssd")) {
                return DetectedObjects.class;
            }
            return Classifications.class;
        }
        return Class.forName(outputClass);
    }

    public Object getInputData() throws IOException, ClassNotFoundException {
        Class<?> klass = getInputClass();
        if (klass == Image.class) {
            return ImageFactory.getInstance().fromFile(getImageFile());
        } else if (klass == float[].class || klass == NDList.class) {
            // TODO: load data from input file
            // Create empty NDArray from shape for now
            return null;
        }
        throw new IllegalArgumentException("Unsupported input class: " + klass);
    }

    public Shape getInputShape() {
        return inputShape;
    }
}
