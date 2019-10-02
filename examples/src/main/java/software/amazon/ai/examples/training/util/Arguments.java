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
package software.amazon.ai.examples.training.util;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import software.amazon.ai.engine.Engine;

public class Arguments {
    private int epoch;
    private int batchSize;
    private int numGpus;
    private boolean isSymbolic;
    private boolean preTrained;

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
}
