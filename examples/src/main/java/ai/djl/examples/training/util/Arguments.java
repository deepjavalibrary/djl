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

import ai.djl.engine.Engine;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;

public class Arguments {
    private int epoch;
    private int batchSize;
    private int numGpus;
    private boolean isSymbolic;
    private boolean preTrained;
    private String outputDir;
    private long maxIterations;

    public Arguments(CommandLine cmd) {
        if (cmd.hasOption("epoch")) {
            epoch = Integer.parseInt(cmd.getOptionValue("epoch"));
        } else {
            epoch = 10;
        }
        if (cmd.hasOption("batch-size")) {
            batchSize = Integer.parseInt(cmd.getOptionValue("batch-size"));
        } else {
            batchSize = 32;
        }
        if (cmd.hasOption("num-gpus")) {
            numGpus = Integer.parseInt(cmd.getOptionValue("num-gpus"));
        } else {
            numGpus = Engine.getInstance().getGpuCount() > 0 ? 1 : 0;
        }
        if (cmd.hasOption("symbolic-model")) {
            isSymbolic = Boolean.parseBoolean(cmd.getOptionValue("symbolic-model"));
        } else {
            isSymbolic = true;
        }
        if (cmd.hasOption("pre-trained")) {
            preTrained = Boolean.parseBoolean(cmd.getOptionValue("pre-trained"));
        } else {
            preTrained = true;
        }
        if (cmd.hasOption("output-dir")) {
            outputDir = cmd.getOptionValue("output-dir");
        } else {
            outputDir = null;
        }
        if (cmd.hasOption("max-iterations")) {
            maxIterations = Long.parseLong(cmd.getOptionValue("max-iterations"));
        } else {
            maxIterations = Long.MAX_VALUE;
        }
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
                        .longOpt("num-gpus")
                        .hasArg()
                        .argName("NUMGPUS")
                        .desc("Number of GPUs used for training")
                        .build());
        options.addOption(
                Option.builder("s")
                        .longOpt("symbolic-model")
                        .hasArg()
                        .argName("SYMBOLIC")
                        .desc("Use symbolic model, use imperative model if false")
                        .build());
        options.addOption(
                Option.builder("p")
                        .longOpt("pre-trained")
                        .hasArg()
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
                        .longOpt("max-iterations")
                        .hasArg()
                        .argName("max-iterations")
                        .desc("Limit each epoch to a fixed number of iterations")
                        .build());
        return options;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public int getEpoch() {
        return epoch;
    }

    public int getNumGpus() {
        return numGpus;
    }

    public boolean getIsSymbolic() {
        return isSymbolic;
    }

    public boolean getPreTrained() {
        return preTrained;
    }

    public String getOutputDir() {
        return outputDir;
    }

    public long getMaxIterations() {
        return maxIterations;
    }
}
