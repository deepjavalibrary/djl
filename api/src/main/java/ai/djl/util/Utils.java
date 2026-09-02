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
import java.net.HttpURLConnection;
import java.net.InetAddress;
import java.net.URL;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileVisitOption;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Scanner;
import java.util.function.Predicate;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/** A class containing utility methods. */
public final class Utils {

    private static final Logger logger = LoggerFactory.getLogger(Utils.class);

    private static final int MAX_REDIRECTS = 5;

    public static final String[] EMPTY_ARRAY = new String[0];

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
        List<Path> list;
        try (Stream<Path> stream = Files.walk(dir)) {
            list = stream.sorted(Comparator.reverseOrder()).collect(Collectors.toList());
        } catch (IOException ignore) {
            return;
        }
        for (Path path : list) {
            try {
                Files.deleteIfExists(path);
            } catch (IOException ignore) {
                // ignore
            }
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
        try (Stream<Path> stream = Files.walk(modelDir, 1, FileVisitOption.FOLLOW_LINKS)) {
            List<Integer> checkpoints =
                    stream.map(
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
     * Returns if offline mode is enabled.
     *
     * @return true if offline mode is enabled
     */
    public static boolean isOfflineMode() {
        String mode = getenv("DJL_OFFLINE", System.getProperty("ai.djl.offline"));
        if (mode != null) {
            return Boolean.parseBoolean(mode);
        }
        // backward compatible
        return Boolean.getBoolean("offline");
    }

    /**
     * Returns nested model directory if the directory contains only one subdirectory.
     *
     * @param modelDir the model directory
     * @return subdirectory if the model directory only contains one subdirectory
     */
    public static Path getNestedModelDir(Path modelDir) {
        if (Files.isDirectory(modelDir)) {
            try (Stream<Path> stream = Files.list(modelDir)) {
                // handle actual model directory is subdirectory case
                List<Path> files =
                        stream.filter(p -> !p.getFileName().toString().startsWith("."))
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
        return getEnvOrSystemProperty(name, null);
    }

    /**
     * Gets the value of the specified environment variable or system property.
     *
     * @param name the name of the environment variable
     * @param def a default value
     * @return the string value of the variable or system property
     */
    public static String getEnvOrSystemProperty(String name, String def) {
        try {
            String env = System.getenv(name);
            if (env != null) {
                return env;
            }
        } catch (SecurityException e) {
            logger.warn("Security manager doesn't allow access to the environment variable");
        }

        String prop = System.getProperty(name);
        if (prop != null) {
            return prop;
        }

        return def;
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
        return openUrl(url, Collections.emptyMap());
    }

    /**
     * Opens a connection to this URL and returns an InputStream for reading from that connection.
     *
     * <p><b>Security Warning:</b> This method should NOT be used with untrusted URL inputs. Using
     * untrusted URLs can lead to:
     *
     * <ul>
     *   <li>Server-Side Request Forgery (SSRF) attacks - allowing access to internal resources
     *   <li>Local file access via file:// protocol
     *   <li>Arbitrary network requests to unintended destinations
     * </ul>
     *
     * <p>Always validate and sanitize URL inputs before passing them to this method. Consider:
     *
     * <ul>
     *   <li>Restricting allowed protocols (e.g., only https://)
     *   <li>Validating against an allowlist of trusted domains
     *   <li>Blocking access to private IP ranges and localhost
     * </ul>
     *
     * @param url the URL to open
     * @param headers HTTP headers
     * @return an input stream for reading from the URL connection.
     * @throws IOException if an I/O exception occurs, or the destination or scheme is not allowed
     */
    public static InputStream openUrl(URL url, Map<String, String> headers) throws IOException {
        // Remote fetches are limited to public http(s) destinations by default, and redirects are
        // resolved explicitly so each hop is checked against the same rule. Local-resource schemes
        // (file:, jar:) are unrestricted, since they do not perform a network fetch. Other schemes
        // are not supported. Set DJL_ALLOW_INSECURE_URL=true (or -Dai.djl.allow_insecure_url=true)
        // to restore the previous unrestricted behavior.
        String protocol = url.getProtocol();
        if ("http".equalsIgnoreCase(protocol) || "https".equalsIgnoreCase(protocol)) {
            return new BufferedInputStream(
                    openHttpConnection(url, "GET", headers).getInputStream());
        }
        // Local-resource schemes and full-opt-out use the legacy direct stream.
        if (isInsecureUrlAllowed()
                || "file".equalsIgnoreCase(protocol)
                || "jar".equalsIgnoreCase(protocol)) {
            return new BufferedInputStream(url.openStream());
        }
        throw new IOException(
                "Blocked request using unsupported URL protocol: "
                        + protocol
                        + ". Set DJL_ALLOW_INSECURE_URL=true to override.");
    }

    /**
     * Opens an http(s) connection using the specified request method, resolving redirects
     * explicitly so that every hop is checked against the same destination rule.
     *
     * <p>This is the shared entry point for remote reads: {@link #openUrl(URL, Map)} uses it for
     * {@code GET}, and callers that only need response metadata can use it for {@code HEAD}. The
     * caller owns the returned connection and should disconnect it when finished.
     *
     * @param url the URL to open
     * @param method the HTTP request method, for example {@code GET} or {@code HEAD}
     * @param headers HTTP headers
     * @return the connection for the final, validated hop
     * @throws IOException if an I/O exception occurs or a hop is not allowed
     */
    public static HttpURLConnection openHttpConnection(
            URL url, String method, Map<String, String> headers) throws IOException {
        if (isOfflineMode()) {
            throw new IOException("Offline mode is enabled.");
        }
        if (isInsecureUrlAllowed()) {
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod(method);
            for (Map.Entry<String, String> entry : headers.entrySet()) {
                conn.addRequestProperty(entry.getKey(), entry.getValue());
            }
            return conn;
        }
        return openHttpConnection(url, method, headers, Utils::isPublicHost);
    }

    /**
     * Opens an http(s) connection, resolving redirects explicitly so that every hop is checked
     * against {@code hostAllowed} rather than being followed by the connection itself.
     *
     * <p>The host check is a parameter so the redirect handling can be exercised against a local
     * test server.
     *
     * @param url the URL to open
     * @param method the HTTP request method
     * @param headers HTTP headers
     * @param hostAllowed the check applied to the host of every hop
     * @return the connection for the final, validated hop
     * @throws IOException if an I/O exception occurs or a hop is not allowed
     */
    static HttpURLConnection openHttpConnection(
            URL url, String method, Map<String, String> headers, Predicate<String> hostAllowed)
            throws IOException {
        URL current = url;
        // One initial request plus at most MAX_REDIRECTS redirect hops.
        for (int i = 0; i <= MAX_REDIRECTS; ++i) {
            String protocol = current.getProtocol();
            if (!"http".equalsIgnoreCase(protocol) && !"https".equalsIgnoreCase(protocol)) {
                throw new IOException("Blocked redirect to unsupported URL protocol: " + protocol);
            }
            if (!hostAllowed.test(current.getHost())) {
                throw new IOException(
                        "Blocked request to non-public address: " + current.getHost());
            }
            HttpURLConnection conn = (HttpURLConnection) current.openConnection();
            conn.setInstanceFollowRedirects(false);
            conn.setRequestMethod(method);
            for (Map.Entry<String, String> entry : headers.entrySet()) {
                conn.addRequestProperty(entry.getKey(), entry.getValue());
            }
            int code;
            try {
                code = conn.getResponseCode();
            } catch (IOException e) {
                conn.disconnect();
                throw e;
            }
            if (code < 300 || code >= 400) {
                return conn;
            }
            // Redirect: re-validate the target on the next loop iteration instead of letting
            // HttpURLConnection follow it blindly (which would defeat the host check above).
            String location = conn.getHeaderField("Location");
            conn.disconnect();
            if (location == null) {
                throw new IOException("Redirect (" + code + ") with no Location header");
            }
            // Resolve relative redirects against the current URL, then re-validate the target.
            current = new URL(current, location);
        }
        throw new IOException("Too many redirects (>" + MAX_REDIRECTS + ") for URL: " + url);
    }

    /**
     * Returns whether insecure (non-public) URL access is explicitly allowed.
     *
     * @return true if {@code DJL_ALLOW_INSECURE_URL} / {@code ai.djl.allow_insecure_url} is set
     *     true
     */
    static boolean isInsecureUrlAllowed() {
        String mode =
                getenv("DJL_ALLOW_INSECURE_URL", System.getProperty("ai.djl.allow_insecure_url"));
        return Boolean.parseBoolean(mode);
    }

    /**
     * Returns whether a host resolves exclusively to public (non-internal) IP addresses.
     *
     * <p>A host is rejected if it is empty or if any resolved address is loopback, link-local,
     * site-local (RFC 1918 private), wildcard, multicast, or an IPv6 unique local address
     * (fc00::/7). The latter is checked explicitly because {@link InetAddress#isSiteLocalAddress()}
     * only covers the deprecated fec0::/10 range. IPv4-mapped IPv6 literals resolve to an {@code
     * Inet4Address} and are covered by the IPv4 checks.
     *
     * <p>Note that this resolves the host name and the subsequent connection resolves it again, so
     * a name that resolves to an allowed address here could resolve to a different address when the
     * connection is made. Pinning the connection to a validated address would be needed to close
     * that gap; environments that require it should restrict egress at the network layer.
     *
     * @param host the host name to validate
     * @return true if every resolved address is a public address
     */
    static boolean isPublicHost(String host) {
        if (host == null || host.isEmpty()) {
            return false;
        }
        try {
            InetAddress[] addresses = InetAddress.getAllByName(host);
            if (addresses.length == 0) {
                return false;
            }
            for (InetAddress addr : addresses) {
                if (addr.isLoopbackAddress()
                        || addr.isLinkLocalAddress()
                        || addr.isSiteLocalAddress()
                        || addr.isAnyLocalAddress()
                        || addr.isMulticastAddress()
                        || isIpv6UniqueLocal(addr)) {
                    return false;
                }
            }
        } catch (UnknownHostException e) {
            return false;
        }
        return true;
    }

    private static boolean isIpv6UniqueLocal(InetAddress addr) {
        byte[] bytes = addr.getAddress();
        // fc00::/7 - the first seven bits are 1111110
        return bytes.length == 16 && (bytes[0] & 0xFE) == 0xFC;
    }

    /**
     * Returns a hash of a string.
     *
     * @param input the input string
     * @return a 20 bytes hash of the input stream in hex format
     */
    public static String hash(String input) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] buf = md.digest(input.getBytes(StandardCharsets.UTF_8));
            return Hex.toHexString(buf, 0, 20);
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError("SHA256 algorithm not found.", e);
        }
    }
}
