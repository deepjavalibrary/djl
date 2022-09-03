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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileVisitOption;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/** A class containing utility methods. */
public final class Utils {

    private static final Logger logger = LoggerFactory.getLogger(Utils.class);

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
     * Renames a file to a target file and ignore error if target already exists.
     *
     * @param source the path to the file to move
     * @param target the path to the target file
     * @throws IOException if move file failed
     */
    public static void moveQuietly(Path source, Path target) throws IOException {
        try {
            Files.move(source, target, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            if (!Files.exists(target)) {
                throw e;
            }
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
        return readLines(file, false);
    }

    /**
     * Reads all lines from a file.
     *
     * @param file the file to be read
     * @param trim true if you want to trim the line and exclude empty lines
     * @return all lines in the file
     * @throws IOException if read file failed
     */
    public static List<String> readLines(Path file, boolean trim) throws IOException {
        if (Files.notExists(file)) {
            return Collections.emptyList();
        }
        try (InputStream is = new BufferedInputStream(Files.newInputStream(file))) {
            return readLines(is, trim);
        }
    }

    /**
     * Reads all lines from the specified InputStream.
     *
     * @param is the InputStream to read
     * @return all lines from the input
     */
    public static List<String> readLines(InputStream is) {
        return readLines(is, false);
    }

    /**
     * Reads all lines from the specified InputStream.
     *
     * @param is the InputStream to read
     * @param trim true if you want to trim the line and exclude empty lines
     * @return all lines from the input
     */
    public static List<String> readLines(InputStream is, boolean trim) {
        List<String> list = new ArrayList<>();
        try (Scanner scanner =
                new Scanner(is, StandardCharsets.UTF_8.name()).useDelimiter("\\n|\\r\\n")) {
            while (scanner.hasNext()) {
                String line = scanner.next();
                if (trim) {
                    line = line.trim();
                    if (line.isEmpty()) {
                        continue;
                    }
                }
                list.add(line);
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
                Files.walk(modelDir, 1, FileVisitOption.FOLLOW_LINKS)
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
        for (Parameter parameter : parameters.values()) {
            logger.debug(
                    "Checking parameter: {} Shape: {}",
                    parameter.getName(),
                    parameter.getArray().getShape());
            checkNDArrayValues(parameter.getArray(), logger, "weight");

            if (parameter.requiresGradient() && checkGradient) {
                logger.debug("Checking gradient of: {}", parameter.getName());
                checkNDArrayValues(parameter.getArray().getGradient(), logger, "grad");
            }
        }
    }

    /**
     * Utility function to help summarize the values in an {@link NDArray}.
     *
     * @param array the {@link NDArray} to be summarized
     * @param logger the logger to log the result
     * @param prefix the prefix or name to be displayed
     */
    public static void checkNDArrayValues(NDArray array, Logger logger, String prefix) {
        if (array.isNaN().any().getBoolean()) {
            logger.warn("There are NANs in value:");
            for (int i = 0; i < array.size(0); i++) {
                logger.warn("{}", array.get(i));
            }
        }
        logger.debug("{} sum: {}", prefix, array.sum().getFloat());
        logger.debug("{} mean: {}", prefix, array.mean().getFloat());
        logger.debug("{} max: {}", prefix, array.max().getFloat());
        logger.debug("{} min: {}", prefix, array.min().getFloat());
        logger.debug("{} shape: {}", prefix, array.getShape().toString());
    }

    /**
     * Utility function to get Engine specific cache directory.
     *
     * @param engine the engine name
     * @return DJL engine cache directory
     */
    public static Path getEngineCacheDir(String engine) {
        return getEngineCacheDir().resolve(engine);
    }

    /**
     * Utility function to get Engine cache directory.
     *
     * @return DJL engine cache directory
     */
    public static Path getEngineCacheDir() {
        String cacheDir = getEnvOrSystemProperty("ENGINE_CACHE_DIR");
        if (cacheDir == null || cacheDir.isEmpty()) {
            return getCacheDir();
        }
        return Paths.get(cacheDir);
    }

    /**
     * Utility function to get DJL cache directory.
     *
     * @return DJL cache directory
     */
    public static Path getCacheDir() {
        String cacheDir = getEnvOrSystemProperty("DJL_CACHE_DIR");
        if (cacheDir == null || cacheDir.isEmpty()) {
            Path dir = Paths.get(System.getProperty("user.home"));
            if (!Files.isWritable(dir)) {
                dir = Paths.get(System.getProperty("java.io.tmpdir"));
            }
            return dir.resolve(".djl.ai");
        }
        return Paths.get(cacheDir);
    }

    /**
     * Returns nested model directory if the directory contains only one subdirectory.
     *
     * @param modelDir the model directory
     * @return subdirectory if the model directory only contains one subdirectory
     */
    public static Path getNestedModelDir(Path modelDir) {
        if (Files.isDirectory(modelDir)) {
            try {
                // handle actual model directory is subdirectory case
                List<Path> files =
                        Files.list(modelDir)
                                .filter(p -> !p.getFileName().toString().startsWith("."))
                                .collect(Collectors.toList());
                if (files.size() == 1 && Files.isDirectory(files.get(0))) {
                    return files.get(0);
                }
            } catch (IOException e) {
                throw new AssertionError("Failed to list files: " + modelDir, e);
            }
        }
        return modelDir.toAbsolutePath();
    }

    /**
     * Gets the value of the specified environment variable or system property.
     *
     * @param name the name of the environment variable
     * @return the string value of the variable or system property
     */
    public static String getEnvOrSystemProperty(String name) {
        return getenv(name, System.getProperty(name));
    }

    /**
     * Gets the value of the specified environment variable.
     *
     * @param name the name of the environment variable
     * @param def a default value
     * @return the string value of the variable, or {@code def} if the variable is not defined in
     *     the system environment or security manager doesn't allow access to the environment
     *     variable
     */
    public static String getenv(String name, String def) {
        try {
            String val = System.getenv(name);
            return val == null ? def : val;
        } catch (SecurityException e) {
            logger.warn("Security manager doesn't allow access to the environment variable");
        }
        return def;
    }

    /**
     * Gets the value of the specified environment variable.
     *
     * @param name the name of the environment variable
     * @return the string value of the variable, or {@code null} if the variable is not defined in
     *     the system environment or security manager doesn't allow access to the environment
     *     variable
     */
    public static String getenv(String name) {
        return getenv(name, null);
    }

    /**
     * Returns an unmodifiable string map view of the current system environment.
     *
     * @return the environment as a map of variable names to values
     */
    public static Map<String, String> getenv() {
        try {
            return System.getenv();
        } catch (SecurityException e) {
            logger.warn("Security manager doesn't allow access to the environment variable");
        }
        return Collections.emptyMap();
    }

    /**
     * Opens a connection to this URL and returns an InputStream for reading from that connection.
     *
     * @param url the url to open
     * @return an input stream for reading from the URL connection.
     * @throws IOException if an I/O exception occurs
     */
    public static InputStream openUrl(String url) throws IOException {
        return openUrl(new URL(url));
    }

    /**
     * Opens a connection to this URL and returns an InputStream for reading from that connection.
     *
     * @param url the url to open
     * @return an input stream for reading from the URL connection.
     * @throws IOException if an I/O exception occurs
     */
    public static InputStream openUrl(URL url) throws IOException {
        String protocol = url.getProtocol();
        if (Boolean.getBoolean("offline")
                && ("http".equalsIgnoreCase(protocol) || "https".equalsIgnoreCase(protocol))) {
            throw new IOException("Offline model is enabled.");
        }
        return new BufferedInputStream(url.openStream());
    }
}
