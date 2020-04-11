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

import java.io.IOException;
import java.util.ArrayList;

public final class ModelExporter {

    private ModelExporter() {}

    public static void processSpawner(String filePath, String pythonPath, Arguments args)
            throws IOException, InterruptedException {
        ArrayList<String> commands = new ArrayList<>();
        commands.add(pythonPath);
        commands.add(filePath);
        commands.addAll(args.getArgs());
        Process p = new ProcessBuilder().command(commands).start();
        p.waitFor();
    }
}
