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
package ai.djl.examples.training.util;

import ai.djl.Device;
import ai.djl.util.JsonUtils;
import com.google.gson.reflect.TypeToken;
import java.lang.reflect.Type;
import java.util.Map;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

public class Arguments {

    private int epoch;
    private int batchSize;
    private int maxGpus;
    private boolean isSymbolic;
    private boolean preTrained;
    private String outputDir;
    private long limit;
    private String modelDir;
    private Map<String, String> criteria;

    public Arguments(CommandLine cmd) {
        if (cmd.hasOption("epoch")) {
            epoch = Integer.parseInt(cmd.getOptionValue("epoch"));
        } else {
            epoch = 2;
        }
        maxGpus = Device.getGpuCount();
        if (cmd.hasOption("max-gpus")) {
            maxGpus = Math.min(Integer.parseInt(cmd.getOptionValue("max-gpus")), maxGpus);
        }
        if (cmd.hasOption("batch-size")) {
            batchSize = Integer.parseInt(cmd.getOptionValue("batch-size"));
        } else {
            batchSize = maxGpus > 0 ? 32 * maxGpus : 32;
        }
        isSymbolic = cmd.hasOption("symbolic-model");
        preTrained = cmd.hasOption("pre-trained");

        if (cmd.hasOption("output-dir")) {
            outputDir = cmd.getOptionValue("output-dir");
        } else {
            outputDir = "build/model";
        }
        if (cmd.hasOption("max-batches")) {
            limit = Long.parseLong(cmd.getOptionValue("max-batches")) * batchSize;
        } else {
            limit = Long.MAX_VALUE;
        }
        if (cmd.hasOption("model-dir")) {
            modelDir = cmd.getOptionValue("model-dir");
        } else {
            modelDir = null;
        }
        if (cmd.hasOption("criteria")) {
            Type type = new TypeToken<Map<String, Object>>() {}.getType();
            criteria = JsonUtils.GSON.fromJson(cmd.getOptionValue("criteria"), type);
        }
    }

    public static Arguments parseArgs(String[] args) throws ParseException {
        Options options = Arguments.getOptions();
        DefaultParser parser = new DefaultParser();
        CommandLine cmd = parser.parse(options, args, null, false);
        return new Arguments(cmd);
    }

    public static Options getOptions() {
        Options options = new Options();
        options.addOption(
                Option.builder("e")
                        .longOpt("epoch")
                        .hasArg()
                        .argName("EPOCH")
                        .desc("Numbers of epochs user would like to run")
                        .build());
        options.addOption(
                Option.builder("b")
                        .longOpt("batch-size")
                        .hasArg()
                        .argName("BATCH-SIZE")
                        .desc("The batch size of the training data.")
                        .build());
        options.addOption(
                Option.builder("g")
                        .longOpt("max-gpus")
                        .hasArg()
                        .argName("MAXGPUS")
                        .desc("Max number of GPUs to use for training")
                        .build());
        options.addOption(
                Option.builder("s")
                        .longOpt("symbolic-model")
                        .argName("SYMBOLIC")
                        .desc("Use symbolic model, use imperative model if false")
                        .build());
        options.addOption(
                Option.builder("p")
                        .longOpt("pre-trained")
                        .argName("PRE-TRAINED")
                        .desc("Use pre-trained weights")
                        .build());
        options.addOption(
                Option.builder("o")
                        .longOpt("output-dir")
                        .hasArg()
                        .argName("OUTPUT-DIR")
                        .desc("Use output to determine directory to save your model parameters")
                        .build());
        options.addOption(
                Option.builder("m")
                        .longOpt("max-batches")
                        .hasArg()
                        .argName("max-batches")
                        .desc(
                                "Limit each epoch to a fixed number of iterations to test the training script")
                        .build());
        options.addOption(
                Option.builder("d")
                        .longOpt("model-dir")
                        .hasArg()
                        .argName("MODEL-DIR")
                        .desc("pre-trained model file directory")
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

    public int getBatchSize() {
        return batchSize;
    }

    public int getEpoch() {
        return epoch;
    }

    public int getMaxGpus() {
        return maxGpus;
    }

    public boolean isSymbolic() {
        return isSymbolic;
    }

    public boolean isPreTrained() {
        return preTrained;
    }

    public String getModelDir() {
        return modelDir;
    }

    public String getOutputDir() {
        return outputDir;
    }

    public long getLimit() {
        return limit;
    }

    public Map<String, String> getCriteria() {
        return criteria;
    }
}
