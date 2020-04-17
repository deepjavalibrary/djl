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

package ai.djl.uploader.util;

import ai.djl.repository.Metadata;
import com.google.gson.Gson;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;

public final class TestUtils {

    private TestUtils() {}

    public static void writeTmpFile(Path dir, String filename, String content) throws IOException {
        Files.createDirectories(dir);
        try (DataOutputStream dos =
                new DataOutputStream(Files.newOutputStream(dir.resolve(filename)))) {
            dos.writeUTF(content);
        }
    }

    public static Metadata readGson(Path path) throws IOException {
        Gson gson = new Gson();
        try (Reader reader = Files.newBufferedReader(path)) {
            return gson.fromJson(reader, Metadata.class);
        }
    }
}
