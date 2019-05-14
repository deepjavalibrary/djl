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

package com.amazon.ai.example.util;

import java.io.File;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

public class Arguments {

    private String modelDir;
    private String modelName;
    private String imageFile;
    private int duration;
    private int iteration = 1000;
    private String question;
    private String answer;
    private int seqLength;
    private String vocabulary;

    public Arguments(CommandLine cmd) {
        modelDir = cmd.getOptionValue("model-dir");
        modelName = cmd.getOptionValue("model-name");
        if (cmd.hasOption("image")) {
            imageFile = cmd.getOptionValue("image");
        }
        if (cmd.hasOption("duration")) {
            duration = Integer.parseInt(cmd.getOptionValue("duration"));
        }
        if (cmd.hasOption("iteration")) {
            iteration = Integer.parseInt(cmd.getOptionValue("iteration"));
        }
        if (cmd.hasOption("question")) {
            question = cmd.getOptionValue("question");
        }
        if (cmd.hasOption("answer")) {
            answer = cmd.getOptionValue("answer");
        }
        if (cmd.hasOption("sequenceLength")) {
            seqLength = Integer.parseInt(cmd.getOptionValue("sequenceLength"));
        }
        if (cmd.hasOption("vocabulary")) {
            vocabulary = cmd.getOptionValue("vocabulary");
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
        options.addOption(
                Option.builder("q")
                        .longOpt("question")
                        .hasArg()
                        .argName("QUESTION")
                        .desc("Question of the model")
                        .build());
        options.addOption(
                Option.builder("a")
                        .longOpt("answer")
                        .hasArg()
                        .argName("ANSWER")
                        .desc("Answer paragraph of the model")
                        .build());
        options.addOption(
                Option.builder("sl")
                        .longOpt("sequenceLength")
                        .hasArg()
                        .argName("SEQUENCELENGTH")
                        .desc("Sequence Length of the paragraph")
                        .build());
        options.addOption(
                Option.builder("v")
                        .longOpt("vocabulary")
                        .hasArg()
                        .argName("VOCABULARY")
                        .desc("Vocabulary of the model")
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

    public String getQuestion() {
        return question;
    }

    public String getAnswer() {
        return answer;
    }

    public int getSeqLength() {
        return seqLength;
    }

    public String getVocabulary() {
        return vocabulary;
    }
}
