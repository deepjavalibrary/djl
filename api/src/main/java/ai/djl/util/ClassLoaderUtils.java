/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import javax.tools.JavaCompiler;
import javax.tools.ToolProvider;

/** A utility class that load classes from specific URLs. */
public final class ClassLoaderUtils {

    private static final Logger logger = LoggerFactory.getLogger(ClassLoaderUtils.class);

    private ClassLoaderUtils() {}

    /**
     * scan classes files from a path to see if there is a matching implementation for a class.
     *
     * <p>For .class file, this function expects them in classes/your/package/ClassName.class
     *
     * @param path the path to scan from
     * @param type the type of the class
     * @param className the name of the classes, pass null if name is unknown
     * @param <T> the Template T for the output Class
     * @return the Class implementation
     */
    public static <T> T findImplementation(Path path, Class<T> type, String className) {
        try {
            Path classesDir = path.resolve("classes");
            // we only consider .class files and skip .java files
            List<Path> jarFiles;
            if (Files.isDirectory(path)) {
                try (Stream<Path> stream = Files.list(path)) {
                    jarFiles =
                            stream.filter(p -> p.toString().endsWith(".jar"))
                                    .collect(Collectors.toList());
                }
            } else {
                jarFiles = Collections.emptyList();
            }
            final URL[] urls = new URL[jarFiles.size() + 1];
            urls[0] = classesDir.toUri().toURL();
            int index = 1;
            for (Path p : jarFiles) {
                urls[index++] = p.toUri().toURL();
            }

            final ClassLoader contextCl = getContextClassLoader();
            ClassLoader cl =
                    AccessController.doPrivileged(
                            (PrivilegedAction<ClassLoader>)
                                    () -> new URLClassLoader(urls, contextCl));
            if (className != null && !className.isEmpty()) {
                T impl = initClass(cl, type, className);
                if (impl == null) {
                    logger.warn("Failed to load class: {}", className);
                }
                return impl;
            }

            T implemented = scanDirectory(cl, type, classesDir);
            if (implemented != null) {
                return implemented;
            }

            for (Path p : jarFiles) {
                implemented = scanJarFile(cl, type, p);
                if (implemented != null) {
                    return implemented;
                }
            }
        } catch (IOException e) {
            logger.debug("Failed to find Translator", e);
        }
        return null;
    }

    private static <T> T scanDirectory(ClassLoader cl, Class<T> type, Path dir) throws IOException {
        if (!Files.isDirectory(dir)) {
            logger.trace("Directory not exists: {}", dir);
            return null;
        }
        try (Stream<Path> stream = Files.walk(dir)) {
            List<Path> files =
                    stream.filter(p -> Files.isRegularFile(p) && p.toString().endsWith(".class"))
                            .collect(Collectors.toList());
            for (Path file : files) {
                Path p = dir.relativize(file);
                String className = p.toString();
                className = className.substring(0, className.lastIndexOf('.'));
                className = className.replace(File.separatorChar, '.');
                T implemented = initClass(cl, type, className);
                if (implemented != null) {
                    return implemented;
                }
            }
        }
        return null;
    }

    private static <T> T scanJarFile(ClassLoader cl, Class<T> type, Path path) throws IOException {
        try (JarFile jarFile = new JarFile(path.toFile())) {
            Enumeration<JarEntry> en = jarFile.entries();
            while (en.hasMoreElements()) {
                JarEntry entry = en.nextElement();
                String fileName = entry.getName();
                if (fileName.endsWith(".class")) {
                    fileName = fileName.substring(0, fileName.lastIndexOf('.'));
                    fileName = fileName.replace('/', '.');
                    T implemented = initClass(cl, type, fileName);
                    if (implemented != null) {
                        return implemented;
                    }
                }
            }
        }
        return null;
    }

    /**
     * Loads the specified class and constructs an instance.
     *
     * @param cl the {@code ClassLoader} to use
     * @param type the type of the class
     * @param className the class to be loaded
     * @param <T> the type of the class
     * @return an instance of the class, null if the class not found
     */
    public static <T> T initClass(ClassLoader cl, Class<T> type, String className) {
        try {
            Class<?> clazz = Class.forName(className, true, cl);
            Class<? extends T> sub = clazz.asSubclass(type);
            Constructor<? extends T> constructor = sub.getConstructor();
            return constructor.newInstance();
        } catch (Throwable e) {
            logger.trace("Not able to load Object", e);
        }
        return null;
    }

    /**
     * Returns the context class loader if available.
     *
     * @return the context class loader if available
     */
    public static ClassLoader getContextClassLoader() {
        ClassLoader cl = Thread.currentThread().getContextClassLoader();
        if (cl == null) {
            return ClassLoaderUtils.class.getClassLoader(); // NOPMD
        }
        return cl;
    }

    /**
     * Finds all the resources with the given name.
     *
     * @param name the resource name
     * @return An enumeration of {@link java.net.URL URL} objects for the resource
     * @throws IOException if I/O errors occur
     */
    public static Enumeration<URL> getResources(String name) throws IOException {
        return getContextClassLoader().getResources(name);
    }

    /**
     * Finds the first resource in class path with the given name.
     *
     * @param name the resource name
     * @return an enumeration of {@link java.net.URL URL} objects for the resource
     * @throws IOException if I/O errors occur
     */
    public static URL getResource(String name) throws IOException {
        Enumeration<URL> en = getResources(name);
        if (en.hasMoreElements()) {
            return en.nextElement();
        }
        return null;
    }

    /**
     * Returns an {@code InputStream} for reading from the resource.
     *
     * @param name the resource name
     * @return an {@code InputStream} for reading
     * @throws IOException if I/O errors occur
     */
    public static InputStream getResourceAsStream(String name) throws IOException {
        URL url = getResource(name);
        if (url == null) {
            throw new IOException("Resource not found in classpath: " + name);
        }
        return url.openStream();
    }

    /**
     * Uses provided nativeHelper to load native library.
     *
     * @param nativeHelper a native helper class that loads native library
     * @param path the native library file path
     */
    public static void nativeLoad(String nativeHelper, String path) {
        try {
            Class<?> clazz = Class.forName(nativeHelper, true, getContextClassLoader());
            Method method = clazz.getDeclaredMethod("load", String.class);
            method.invoke(null, path);
        } catch (ReflectiveOperationException e) {
            throw new IllegalArgumentException("Invalid native_helper: " + nativeHelper, e);
        }
    }

    /**
     * Tries to compile java classes in the directory.
     *
     * <p>Compiling bundled {@code .java} sources at model-load time is disabled by default. Enable
     * it with {@code DJL_COMPILE_JAVA=true} or {@code -Dai.djl.compile_java=true} when loading
     * models from a trusted source. Models that ship precompiled {@code .class} or {@code .jar}
     * files, or that supply a translator programmatically, are unaffected.
     *
     * @param dir the directory to scan java file.
     */
    public static void compileJavaClass(Path dir) {
        String[] files = findJavaSources(dir);
        if (files.length == 0) {
            // Nothing to compile: the common case of a model without bundled sources stays quiet.
            return;
        }
        if (!isDynamicCompilationEnabled()) {
            // Skipped rather than failed here: whether this matters depends on the caller, since a
            // model may also ship a precompiled .class or .jar that satisfies the lookup. Callers
            // that resolve an implementation from the output report it via hasSkippedJavaSources
            // when the lookup finds nothing, so a load cannot silently change behavior.
            logger.warn(
                    "Skipping {} bundled java file(s) in {}: compiling bundled sources is disabled"
                            + " by default. Set DJL_COMPILE_JAVA=true or"
                            + " -Dai.djl.compile_java=true to enable it for models from a trusted"
                            + " source, or ship precompiled .class/.jar files instead.",
                    files.length,
                    dir);
            return;
        }
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        if (compiler == null) {
            logger.warn(
                    "Cannot compile {} bundled java file(s) in {}: no system java compiler is"
                            + " available, a JDK is required.",
                    files.length,
                    dir);
            return;
        }
        try {
            int rc = compiler.run(null, null, null, files);
            if (rc != 0) {
                // Report the failure: the caller resolves a translator from the compiled output, so
                // a failed compile otherwise surfaces later as a missing class or as a silent
                // fallback to a default implementation. Compiler diagnostics go to stderr.
                logger.error(
                        "Failed to compile {} bundled java file(s) in {}, compiler exit code {}."
                                + " See the compiler output for details.",
                        files.length,
                        dir,
                        rc);
            }
        } catch (Throwable e) {
            logger.warn("Failed to compile bundled java file", e);
        }
    }

    /**
     * Returns whether {@code dir} holds bundled {@code .java} sources that were not compiled
     * because compilation is disabled.
     *
     * <p>Callers that resolve an implementation from the compiled output use this to report a
     * missing implementation as a configuration problem rather than falling back silently. It is
     * false once the opt-in is set, and false when the directory holds no sources.
     *
     * @param dir the directory that was scanned for java sources
     * @return true if sources are present and compilation is disabled
     */
    public static boolean hasSkippedJavaSources(Path dir) {
        return !isDynamicCompilationEnabled() && findJavaSources(dir).length > 0;
    }

    private static String[] findJavaSources(Path dir) {
        try {
            if (!Files.isDirectory(dir)) {
                logger.debug("Directory not exists: {}", dir);
                return Utils.EMPTY_ARRAY;
            }
            try (Stream<Path> stream = Files.walk(dir)) {
                return stream.filter(p -> Files.isRegularFile(p) && p.toString().endsWith(".java"))
                        .map(p -> p.toAbsolutePath().toString())
                        .toArray(String[]::new);
            }
        } catch (IOException | RuntimeException e) {
            // Files.walk reports traversal errors from the terminal operation as an unchecked
            // UncheckedIOException (an unreadable subdirectory, a symlink loop, a directory removed
            // concurrently), so catching IOException alone would turn a scan problem into a failed
            // model load.
            logger.warn("Failed to scan for bundled java files in {}", dir, e);
            return Utils.EMPTY_ARRAY;
        }
    }

    /**
     * Returns whether dynamic compilation of bundled java sources is explicitly enabled.
     *
     * @return true if {@code DJL_COMPILE_JAVA} / {@code ai.djl.compile_java} is set true
     */
    static boolean isDynamicCompilationEnabled() {
        String mode = Utils.getenv("DJL_COMPILE_JAVA", System.getProperty("ai.djl.compile_java"));
        return Boolean.parseBoolean(mode);
    }
}
