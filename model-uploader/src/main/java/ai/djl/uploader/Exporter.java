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

package ai.djl.uploader;

import ai.djl.uploader.arguments.Arguments;
import ai.djl.uploader.arguments.GluonCvArgs;
import ai.djl.uploader.arguments.HuggingFaceArgs;
import ai.djl.uploader.arguments.KerasArgs;
import java.io.IOException;
import java.io.InputStream;
import java.io.InterruptedIOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

public final class Exporter {

    private Exporter() {}

    public static void main(String[] args) throws IOException, InterruptedException {
        Options options = ExporterArguments.getOptions();
        try {
            DefaultParser parser = new DefaultParser();
            CommandLine cmd = parser.parse(options, args, null, false);
            ExporterArguments arguments = new ExporterArguments(cmd);
            String category = arguments.getCategory();
            MetadataBuilder metadataBuilder;
            if ("dataset".equals(category)) {
                metadataBuilder = dataset(arguments);
            } else if ("gluoncv".equals(category)) {
                metadataBuilder = gluonCvImport(arguments);
            } else if ("keras".equals(category)) {
                metadataBuilder = kerasImport(arguments);
            } else if ("huggingface".equals(category)) {
                metadataBuilder = huggingfaceImport(arguments);
            } else {
                metadataBuilder = readyToGoModel(arguments);
            }
            if (arguments.isRemote()) {
                metadataBuilder.buildExternal();
            } else {
                metadataBuilder.buildLocal();
            }
        } catch (ParseException e) {
            HelpFormatter formatter = new HelpFormatter();
            formatter.setLeftPadding(1);
            formatter.setWidth(120);
            formatter.printHelp(e.getMessage(), options);
        }
    }

    public static MetadataBuilder gluonCvImport(ExporterArguments arguments)
            throws IOException, InterruptedException {
        GluonCvArgs args = new GluonCvArgs();
        args.setName(arguments.getArtifactName());
        args.setOutputPath(arguments.getBaseDir());
        args.setShape(arguments.getShape());
        GluonCvMetaBuilder exporter =
                new GluonCvMetaBuilder()
                        .setApplication(arguments.getApplication())
                        .setArgs(args)
                        .setBaseDir(arguments.getBaseDir())
                        .setArtifactId(arguments.getArtifactId())
                        .setName(arguments.getName())
                        .optProperties(arguments.getProperties())
                        .optArguments(arguments.getArguments())
                        .setDescription(arguments.getDescription());
        String pythonPath = arguments.getPythonPath();
        if (pythonPath != null) {
            exporter.optPythonPath(pythonPath);
        }
        return exporter.prepareBuild();
    }

    public static MetadataBuilder huggingfaceImport(ExporterArguments arguments)
            throws IOException, InterruptedException {
        HuggingFaceArgs args = new HuggingFaceArgs();
        args.setName(arguments.getArtifactName());
        args.setApplicationName(arguments.getApplicationType());
        args.setOutputPath(arguments.getBaseDir());
        args.setShape(arguments.getShape());
        HuggingFaceMetaBuilder exporter =
                new HuggingFaceMetaBuilder()
                        .setApplication(arguments.getApplication())
                        .setArgs(args)
                        .setBaseDir(arguments.getBaseDir())
                        .setArtifactId(arguments.getArtifactId())
                        .setName(arguments.getName())
                        .optProperties(arguments.getProperties())
                        .optArguments(arguments.getArguments())
                        .setDescription(arguments.getDescription());
        String pythonPath = arguments.getPythonPath();
        if (pythonPath != null) {
            exporter.optPythonPath(pythonPath);
        }
        return exporter.prepareBuild();
    }

    public static MetadataBuilder kerasImport(ExporterArguments arguments)
            throws IOException, InterruptedException {
        KerasArgs args = new KerasArgs();
        Path artifactDir;
        try {
            artifactDir = arguments.getArtifactDir();
        } catch (IOException e) {
            // case load by name
            artifactDir = Paths.get(arguments.getBaseDir());
        }
        args.setArtifactPath(artifactDir);
        args.setModelName(arguments.getArtifactName());
        KerasMetaBuilder exporter =
                new KerasMetaBuilder()
                        .setApplication(arguments.getApplication())
                        .setArgs(args)
                        .setBaseDir(arguments.getBaseDir())
                        .setArtifactId(arguments.getArtifactId())
                        .setArtifactName(arguments.getArtifactName())
                        .setName(arguments.getName())
                        .setDescription(arguments.getDescription());
        String pythonPath = arguments.getPythonPath();
        if (pythonPath != null) {
            exporter.optPythonPath(pythonPath);
        }
        return exporter.prepareBuild();
    }

    public static MetadataBuilder readyToGoModel(ExporterArguments arguments) throws IOException {
        return MetadataBuilder.builder()
                .setName(arguments.getName())
                .setDescription(arguments.getDescription())
                .setApplication(arguments.getApplication())
                .setArtifactDir(arguments.getArtifactDir())
                .setArtifactId(arguments.getArtifactId())
                .setBaseDir(arguments.getBaseDir())
                .setGroupId(arguments.getGroupId())
                .setArtifactName(arguments.getArtifactName())
                .addProperties(arguments.getProperties())
                .addArguments(new LinkedHashMap<>(arguments.getArguments()));
    }

    public static MetadataBuilder dataset(ExporterArguments arguments) throws IOException {
        MetadataBuilder builder = readyToGoModel(arguments);
        builder.optIsDataset();
        return builder;
    }

    public static void processSpawner(String filePath, String pythonPath, Arguments args)
            throws IOException, InterruptedException {
        ArrayList<String> commands = new ArrayList<>();
        commands.add(pythonPath);
        commands.add(filePath);
        commands.addAll(args.getArgs());
        Process p = new ProcessBuilder().command(commands).start();
        int exitCode = p.waitFor();
        if (exitCode != 0) {
            InputStream errorStream = p.getErrorStream();
            int c;
            StringBuilder result = new StringBuilder();
            while ((c = errorStream.read()) != -1) {
                result.append((char) c);
            }
            throw new InterruptedIOException(result.toString());
        }
        p.destroy();
    }
}
