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
package ai.djl.uploader.arguments;

import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class HuggingFaceArgs implements Arguments {

    private String name;
    private String applicationName;
    private String outputPath;
    private String shape;

    public void setName(String name) {
        this.name = name;
    }

    public void setApplicationName(String applicationName) {
        this.applicationName = applicationName;
    }

    public void setOutputPath(String outputPath) {
        this.outputPath = Paths.get(outputPath).resolve("transformers").toAbsolutePath().toString();
    }

    public void setShape(String shape) {
        this.shape = shape;
    }

    public String getOutputPath() {
        return outputPath;
    }

    public String getShape() {
        return shape;
    }

    public String getName() {
        return name;
    }

    @Override
    public List<String> getArgs() {
        List<String> args = new ArrayList<>();
        args.add("--name");
        args.add(name);
        args.add("--model_application");
        args.add(applicationName);
        args.add("--output_path");
        args.add(outputPath);
        args.add("--shape");
        args.add(shape);
        return args;
    }
}
