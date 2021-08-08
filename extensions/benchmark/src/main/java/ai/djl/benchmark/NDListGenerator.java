/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import java.io.BufferedOutputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A class generates NDList files. */
final class NDListGenerator {

    private static final Logger logger = LoggerFactory.getLogger(NDListGenerator.class);

    private NDListGenerator() {}

    static boolean generate(String[] args) {
        Options options = getOptions();
        try {
            if (Arguments.hasHelp(args)) {
                Arguments.printHelp(
                        "usage: djl-bench ndlist-gen -s INPUT-SHAPES -o OUTPUT_FILE", options);
                return true;
            }
            DefaultParser parser = new DefaultParser();
            CommandLine cmd = parser.parse(options, args, null, false);
            String inputShapes = cmd.getOptionValue("input-shapes");
            String output = cmd.getOptionValue("output-file");
            boolean ones = cmd.hasOption("ones");
            Path path = Paths.get(output);

            try (NDManager manager = NDManager.newBaseManager()) {
                NDList list = new NDList();
                for (Pair<DataType, Shape> pair : parseShape(inputShapes)) {
                    DataType dataType = pair.getKey();
                    Shape shape = pair.getValue();
                    if (ones) {
                        list.add(manager.ones(shape, dataType));
                    } else {
                        list.add(manager.zeros(shape, dataType));
                    }
                }
                try (OutputStream os = new BufferedOutputStream(Files.newOutputStream(path))) {
                    list.encode(os);
                }
            }
            logger.info("NDList file created: {}", path.toAbsolutePath());
            return true;
        } catch (ParseException e) {
            Arguments.printHelp(e.getMessage(), options);
        } catch (Throwable t) {
            logger.error("Unexpected error", t);
        }
        return false;
    }

    static PairList<DataType, Shape> parseShape(String shape) {
        PairList<DataType, Shape> inputShapes = new PairList<>();
        if (shape != null) {
            if (shape.contains("(")) {
                Pattern pattern =
                        Pattern.compile("\\((\\s*(\\d+)([,\\s]+\\d+)*\\s*)\\)([sdubilBfS]?)");
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
        return inputShapes;
    }

    private static Options getOptions() {
        Options options = new Options();
        options.addOption(
                Option.builder("h").longOpt("help").hasArg(false).desc("Print this help.").build());
        options.addOption(
                Option.builder("s")
                        .required()
                        .longOpt("input-shapes")
                        .hasArg()
                        .argName("INPUT-SHAPES")
                        .desc("Input data shapes for the model.")
                        .build());
        options.addOption(
                Option.builder("o")
                        .required()
                        .longOpt("output-file")
                        .hasArg()
                        .argName("OUTPUT-FILE")
                        .desc("Write output NDList to file.")
                        .build());
        options.addOption(
                Option.builder("1")
                        .longOpt("ones")
                        .hasArg(false)
                        .argName("ones")
                        .desc("Use all ones instead of zeros.")
                        .build());
        return options;
    }
}
