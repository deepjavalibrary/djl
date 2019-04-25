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
package org.apache.mxnet.jna;

import com.sun.jna.Native;
import com.sun.jna.Platform;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Properties;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

final class LibUtils {

    private static final Logger logger = LoggerFactory.getLogger(LibUtils.class);

    private static final String LIB_NAME = "mxnet";
    private static final Pattern PATH_PATTERN = Pattern.compile("\\s*'(.+)',");

    private LibUtils() {}

    public static MxnetLibrary loadLibrary() {
        String libName = LibUtils.findLibraryInClasspath();
        if (libName == null) {
            libName = LibUtils.findLibraryFromSystem();
            if (libName == null) {
                libName = LIB_NAME;
            }
        }
        logger.debug("Loading mxnet library from: {}", libName);

        return (MxnetLibrary) Native.loadLibrary(libName, MxnetLibrary.class);
    }

    private static synchronized String findLibraryInClasspath() {
        URL url = LibUtils.class.getResource("/native/lib/mxnet.properties");
        if (url == null) {
            return null;
        }

        File tmp = null;
        try (InputStream conf = url.openStream()) {
            Properties prop = new Properties();
            prop.load(conf);
            String version = prop.getProperty("version");
            String libs = prop.getProperty("libraries");
            String[] files = libs.split(",");

            String userHome = System.getProperty("user.home");
            String libName = System.mapLibraryName(LIB_NAME);
            File dir = new File(userHome, ".mxnet/cache/" + version);
            File path = new File(dir, libName);
            if (path.exists()) {
                return path.getAbsolutePath();
            }
            tmp = new File(userHome, ".mxnet/cache/tmp");
            Files.createDirectories(tmp.toPath());
            for (String file : files) {
                String libPath = "/native/lib/" + file;
                try (InputStream is = LibUtils.class.getResourceAsStream(libPath);
                        OutputStream os = new FileOutputStream(new File(tmp, file))) {
                    IOUtils.copy(is, os);
                }
            }
            Files.move(tmp.toPath(), dir.toPath());
            tmp = null;
            return path.getAbsolutePath();
        } catch (IOException e) {
            logger.error("Failed to load mxnet native library.", e);
            return null;
        } finally {
            if (tmp != null) {
                FileUtils.deleteQuietly(tmp);
            }
        }
    }

    private static String findLibraryFromSystem() {
        String libPath = System.getProperty("java.library.path");
        if (libPath != null) {
            String[] paths = libPath.split(File.pathSeparator);
            List<String> mappedLibNames;
            if (Platform.isMac()) {
                mappedLibNames = Arrays.asList("libmxnet.dylib", "libmxnet.jnilib", "libmxnet.so");
            } else {
                mappedLibNames = Collections.singletonList(System.mapLibraryName(LIB_NAME));
            }

            for (String path : paths) {
                for (String name : mappedLibNames) {
                    File file = new File(path, name);
                    if (file.exists()) {
                        return file.getAbsolutePath();
                    }
                }
            }
        }

        String libName = searchPythonPath("python3 -m site");
        if (libName != null) {
            return libName;
        }
        return searchPythonPath("python -m site");
    }

    private static String searchPythonPath(String cmd) {
        String libName;
        if (Platform.isMac()) {
            // pip package use libmxnet.so instead of libmxnet.dylib, JNA by default only
            // load .dylib file, we have to use absolute path to load libmxnet.so
            libName = "libmxnet.so";
        } else {
            libName = System.mapLibraryName(LIB_NAME);
        }

        try {
            Process process = Runtime.getRuntime().exec(cmd);
            try (BufferedReader reader =
                    new BufferedReader(
                            new InputStreamReader(
                                    process.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    Matcher m = PATH_PATTERN.matcher(line);
                    if (m.matches()) {
                        File dir = new File(m.group(1));
                        if (dir.isDirectory()) {
                            File file = new File(dir, "mxnet/" + libName);
                            if (file.exists()) {
                                return file.getAbsolutePath();
                            }
                        }
                    }
                }
            }
        } catch (IOException e) {
            if (logger.isTraceEnabled()) {
                logger.trace("Failed execute cmd: " + cmd, e);
            }
        }
        return null;
    }
}
