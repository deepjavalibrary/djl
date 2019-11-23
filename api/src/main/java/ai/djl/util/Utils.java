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
package ai.djl.util;

import ai.djl.ndarray.NDArray;
import ai.djl.nn.Parameter;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.slf4j.Logger;

/** A class containing utility methods. */
public final class Utils {

    private Utils() {}

    /**
     * Returns the index of the first occurrence of the specified element in {@code array}, or -1 if
     * this list does not contain the element.
     *
     * @param array the input array
     * @param value the element to search for
     * @param <T> the array type
     * @return the index of the first occurrence of the specified element in {@code array}, or -1 if
     *     this list does not contain the element
     */
    public static <T> int indexOf(T[] array, T value) {
        if (array != null) {
            for (int i = 0; i < array.length; ++i) {
                if (value.equals(array[i])) {
                    return i;
                }
            }
        }

        return -1;
    }

    /**
     * Returns {@code true} if the {@code array} contains the specified element.
     *
     * @param array the input array
     * @param value the element whose presence in {@code array} is to be tested
     * @param <T> the array type
     * @return {@code true} if this list contains the specified element
     */
    public static <T> boolean contains(T[] array, T value) {
        return indexOf(array, value) >= 0;
    }

    /**
     * Adds padding chars to specified StringBuilder.
     *
     * @param sb the StringBuilder to append
     * @param c the padding char
     * @param count the number characters to be added
     */
    public static void pad(StringBuilder sb, char c, int count) {
        for (int i = 0; i < count; ++i) {
            sb.append(c);
        }
    }

    /**
     * Deletes an entire directory and ignore all errors.
     *
     * @param dir the directory to be removed
     */
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

    /**
     * Reads {@code is} as UTF-8 string.
     *
     * @param is the InputStream to be read
     * @return a UTF-8 encoded string
     * @throws IOException if IO error occurs
     */
    public static String toString(InputStream is) throws IOException {
        return new String(toByteArray(is), StandardCharsets.UTF_8.name());
    }

    /**
     * Reads {@code is} as byte array.
     *
     * @param is the InputStream to be read
     * @return a byte array
     * @throws IOException if IO error occurs
     */
    public static byte[] toByteArray(InputStream is) throws IOException {
        byte[] buf = new byte[81920];
        int read;
        ByteArrayOutputStream bos = new ByteArrayOutputStream(81920);
        while ((read = is.read(buf)) != -1) {
            bos.write(buf, 0, read);
        }
        bos.close();
        return bos.toByteArray();
    }

    /**
     * Reads all lines from a file.
     *
     * @param file the file to be read
     * @return all lines in the file
     * @throws IOException if read file failed
     */
    public static List<String> readLines(Path file) throws IOException {
        if (Files.notExists(file)) {
            return Collections.emptyList();
        }
        try (InputStream is = Files.newInputStream(file)) {
            return readLines(is);
        }
    }

    /**
     * Reads all lines from the specified InputStream.
     *
     * @param is the InputStream to read
     * @return all lines from the input
     */
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

    /**
     * Converts a List of Number to float array.
     *
     * @param list the list to be converted
     * @return a float array
     */
    public static float[] toFloatArray(List<? extends Number> list) {
        float[] ret = new float[list.size()];
        int idx = 0;
        for (Number n : list) {
            ret[idx++] = n.floatValue();
        }
        return ret;
    }

    /**
     * Gets the current epoch number.
     *
     * @param modelDir the path to the directory where the model files are stored
     * @param modelName the name of the model
     * @return the current epoch number
     * @throws IOException if an I/O error occurs
     */
    public static int getCurrentEpoch(Path modelDir, String modelName) throws IOException {
        final Pattern pattern = Pattern.compile(Pattern.quote(modelName) + "-(\\d{4}).params");
        List<Integer> checkpoints =
                Files.walk(modelDir, 1)
                        .map(
                                p -> {
                                    Matcher m = pattern.matcher(p.toFile().getName());
                                    if (m.matches()) {
                                        return Integer.parseInt(m.group(1));
                                    }
                                    return null;
                                })
                        .filter(Objects::nonNull)
                        .sorted()
                        .collect(Collectors.toList());
        if (checkpoints.isEmpty()) {
            return -1;
        }
        return checkpoints.get(checkpoints.size() - 1);
    }

    /**
     * Utility function to help debug nan values in parameters and their gradients.
     *
     * @param parameters the list of parameters to check
     * @param checkGradient whether to check parameter value or its gradient value
     * @param logger the logger to log the result
     */
    public static void checkParameterValues(
            PairList<String, Parameter> parameters, boolean checkGradient, Logger logger) {
        List<Float> values = new ArrayList<>();
        String valueName = checkGradient ? "gradient" : "value";

        values.addAll(
                parameters
                        .values()
                        .stream()
                        .filter(Parameter::requireGradient)
                        .map(
                                param -> {
                                    NDArray value =
                                            checkGradient
                                                    ? param.getArray().getGradient()
                                                    : param.getArray();
                                    float[] sums = value.sum().toFloatArray();
                                    float sum = 0f;
                                    for (float num : sums) {
                                        sum += num;
                                    }
                                    if (Float.isNaN(sum)) {
                                        logger.info(
                                                "param name: "
                                                        + param.getName()
                                                        + ", "
                                                        + valueName
                                                        + " is nan :"
                                                        + sum);
                                    }
                                    return sum;
                                })
                        .collect(Collectors.toList()));

        logger.debug(
                "Sum of param's"
                        + valueName
                        + "is : "
                        + values.stream().mapToDouble(f -> f.doubleValue()).sum());
    }
}
