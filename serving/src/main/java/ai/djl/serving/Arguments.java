/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving;

import ai.djl.serving.util.ConfigManager;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;

/** A class represents parsed command line arguments. */
public final class Arguments {

    private String configFile;
    private String modelStore;
    private String[] models;
    private boolean help;

    /**
     * Constructs a new {@code Arguments} instance.
     *
     * @param cmd a parsed {@code CommandLine}
     */
    public Arguments(CommandLine cmd) {
        configFile = cmd.getOptionValue("config-file");
        modelStore = cmd.getOptionValue("model-store");
        models = cmd.getOptionValues("models");
        help = cmd.hasOption("help");
    }

    /**
     * Builds the command line options.
     *
     * @return the command line options
     */
    public static Options getOptions() {
        Options options = new Options();
        options.addOption(
                Option.builder("h").longOpt("help").hasArg(false).desc("Print this help.").build());
        options.addOption(
                Option.builder("f")
                        .longOpt("config-file")
                        .hasArg()
                        .argName("CONFIG-FILE")
                        .desc("Path to the configuration properties file.")
                        .build());
        options.addOption(
                Option.builder("m")
                        .longOpt("models")
                        .hasArgs()
                        .argName("MODELS")
                        .desc("Models to be loaded at startup.")
                        .build());
        options.addOption(
                Option.builder("s")
                        .longOpt("model-store")
                        .hasArg()
                        .argName("MODELS-STORE")
                        .desc("Model store location where models can be loaded.")
                        .build());
        return options;
    }

    /**
     * Returns the configuration file path.
     *
     * @return the configuration file path
     */
    public Path getConfigFile() {
        if (configFile == null) {
            configFile = System.getProperty("ai.djl.conf", null);
        }
        if (configFile != null) {

            Path file = Paths.get(configFile);
            if (!Files.isRegularFile(file)) {
                throw new IllegalArgumentException("Configuration file not found: " + configFile);
            }
            return file;
        }
        Path modelServerHome = Paths.get(ConfigManager.getModelServerHome());
        Path file = modelServerHome.resolve("conf/config.properties");
        if (Files.isRegularFile(file)) {
            return file;
        }
        file = modelServerHome.resolve("config.properties");
        if (Files.isRegularFile(file)) {
            return file;
        }
        return null;
    }

    /**
     * Returns the model store location.
     *
     * @return the model store location
     */
    public String getModelStore() {
        return modelStore;
    }

    /**
     * Returns the model urls that specified in command line.
     *
     * @return the model urls that specified in command line
     */
    public String[] getModels() {
        return models;
    }

    /**
     * Returns if the command line has help option.
     *
     * @return {@code true} if the command line has help option
     */
    public boolean hasHelp() {
        return help;
    }
}
