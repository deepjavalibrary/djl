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

import ai.djl.ndarray.types.Shape;
import ai.djl.uploader.Arguments;
import java.util.ArrayList;
import java.util.List;

public final class GluonCvArgs implements Arguments {

    private String name;
    private String outputPath;
    private Shape shape;

    public void setName(String name) {
        this.name = name;
    }

    public void setOutputPath(String outputPath) {
        this.outputPath = outputPath;
    }

    public void setShape(Shape shape) {
        this.shape = shape;
    }

    public String getName() {
        return name;
    }

    public Shape getShape() {
        return shape;
    }

    public String getOutputPath() {
        return outputPath;
    }

    @Override
    public List<String> getArgs() {
        List<String> args = new ArrayList<>();
        args.add("--name");
        args.add(name);
        args.add("--output_path");
        args.add(outputPath);
        args.add("--shape");
        args.add(shape.toString());
        return args;
    }
}
