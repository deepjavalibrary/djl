package org.apache.mxnet; /*
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

import java.io.File;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

class Arguments {

    private String modelDir;
    private String modelName;
    private String imageFile;
    private int duration;
    private int iteration = 1000;

    public Arguments(CommandLine cmd) {
        modelDir = cmd.getOptionValue("model-dir");
        modelName = cmd.getOptionValue("model-name");
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
        options.addOption(
                Option.builder("p")
                        .longOpt("model-dir")
                        .required()
                        .hasArg()
                        .argName("MODEL-DIR")
                        .desc("Path to the model directory.")
                        .build());
        options.addOption(
                Option.builder("n")
                        .longOpt("model-name")
                        .required()
                        .hasArg()
                        .argName("MODEL-NAME")
                        .desc("Model name prefix.")
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
        return options;
    }

    public String getModelDir() {
        return modelDir;
    }

    public String getModelName() {
        return modelName;
    }

    public String getImageFile() throws ParseException {
        if (imageFile == null) {
            File file = new File(modelDir, "kitten.jpg");
            if (file.exists()) {
                return file.getAbsolutePath();
            } else {
                throw new ParseException("Missing --image parameter.");
            }
        }
        return imageFile;
    }

    public int getDuration() {
        return duration;
    }

    public int getIteration() {
        return iteration;
    }
}
