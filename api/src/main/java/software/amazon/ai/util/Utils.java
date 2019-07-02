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
package software.amazon.ai.util;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Scanner;
import software.amazon.ai.ndarray.types.DataType;

public final class Utils {

    public static final boolean DEBUG = true;

    private Utils() {}

    public static <T> int indexOf(T[] arr, T value) {
        if (arr != null) {
            for (int i = 0; i < arr.length; ++i) {
                if (value.equals(arr[i])) {
                    return i;
                }
            }
        }

        return -1;
    }

    public static <T> boolean contains(T[] arr, T value) {
        return indexOf(arr, value) >= 0;
    }

    public static void pad(StringBuilder sb, char c, int count) {
        for (int i = 0; i < count; ++i) {
            sb.append(c);
        }
    }

    public static void deleteQuietly(Path dir) {
        try {
            Files.walk(dir)
                    .sorted(Comparator.reverseOrder())
                    .forEach(
                            path -> {
                                try {
                                    Files.deleteIfExists(path);
                                } catch (IOException ignore) {
                                    // ignore
                                }
                            });
        } catch (IOException ignore) {
            // ignore
        }
    }

    public static List<String> readLines(Path file) throws IOException {
        if (Files.notExists(file)) {
            return Collections.emptyList();
        }
        try (InputStream is = Files.newInputStream(file)) {
            return readLines(is);
        }
    }

    public static List<String> readLines(InputStream is) {
        List<String> list = new ArrayList<>();
        try (Scanner scanner =
                new Scanner(is, StandardCharsets.UTF_8.name()).useDelimiter("\\n|\\r\\n")) {
            while (scanner.hasNext()) {
                list.add(scanner.next());
            }
        }
        return list;
    }

    public static CharSequence toCharSequence(ByteBuffer buf, DataType dataType) {
        StringBuilder sb = new StringBuilder();
        while (buf.hasRemaining()) {
            switch (dataType) {
                case FLOAT32:
                    sb.append(String.format("%.8e", buf.getFloat()));
                    break;
                case FLOAT64:
                    sb.append(String.format("%.8e", buf.getDouble()));
                    break;
                case INT8:
                    sb.append(buf.get());
                    break;
                case INT32:
                    sb.append(buf.getInt());
                    break;
                case INT64:
                    sb.append(buf.getLong());
                    break;
                default:
                    throw new IllegalStateException("Unsupported DataType: " + dataType);
            }
            if (buf.hasRemaining()) {
                sb.append(", ");
            }
        }
        return sb;
    }

    public static float[] toFloatArray(List<? extends Number> list) {
        float[] ret = new float[list.size()];
        int idx = 0;
        for (Number n : list) {
            ret[idx++] = n.floatValue();
        }
        return ret;
    }
}
