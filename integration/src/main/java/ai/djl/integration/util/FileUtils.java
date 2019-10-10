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
package ai.djl.integration.util;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

public final class FileUtils {

    private FileUtils() {}

    public static void download(String source, Path destination, String fileName)
            throws IOException {
        URL url = new URL(source);
        try (InputStream in = url.openStream()) {
            if (!Files.exists(destination)) {
                Files.createDirectories(destination);
            }
            Files.copy(in, destination.resolve(fileName), StandardCopyOption.REPLACE_EXISTING);
        }
    }
}
