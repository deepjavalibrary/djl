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
package ai.djl.training.util;

import org.apache.commons.io.FileUtils;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/** A utility class downloads the file from specified url. */
public final class DownloadUtils {

    private DownloadUtils() {}

    /**
     * Downloads a file from specified url.
     *
     * @param url the url to download
     * @param output the output location
     * @throws IOException when IO operation fails in downloading
     */
    public static void download(String url, String output) throws IOException {
        download(new URL(url.trim()), Paths.get(output.trim()));
    }

    /**
     * Downloads a file from specified url.
     *
     * @param url the url to download
     * @param output the output location
     * @throws IOException when IO operation fails in downloading
     */
    public static void download(URL url, Path output) throws IOException {
        if (Files.exists(output)) {
            return;
        }
        FileUtils.copyURLToFile(url, output.toFile());
    }
}
