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

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Collection;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
     * @param className the name of the classes, pass null if name is unknown
     * @param <T> the Template T for the output Class
     * @return the Class implementation
     */
    public static <T> T findImplementation(Path path, String className) {
        try {
            Path classesDir = path.resolve("classes");
            // we only consider .class files and skip .java files
            List<Path> jarFiles;
            if (Files.isDirectory(path)) {
                jarFiles =
                        Files.list(path)
                                .filter(p -> p.toString().endsWith(".jar"))
                                .collect(Collectors.toList());
            } else {
                jarFiles = Collections.emptyList();
            }
            final URL[] urls = new URL[jarFiles.size() + 1];
            urls[0] = classesDir.toUri().toURL();
            int index = 1;
            for (Path p : jarFiles) {
                urls[index++] = p.toUri().toURL();
            }

            final ClassLoader contextCl = Thread.currentThread().getContextClassLoader();
            ClassLoader cl =
                    AccessController.doPrivileged(
                            (PrivilegedAction<ClassLoader>)
                                    () -> new URLClassLoader(urls, contextCl));
            if (className != null && !className.isEmpty()) {
                return initClass(cl, className);
            }

            T implemented = scanDirectory(cl, classesDir);
            if (implemented != null) {
                return implemented;
            }

            for (Path p : jarFiles) {
                implemented = scanJarFile(cl, p);
                if (implemented != null) {
                    return implemented;
                }
            }
        } catch (IOException e) {
            logger.debug("Failed to find Translator", e);
        }
        return null;
    }

    private static <T> T scanDirectory(ClassLoader cl, Path dir) throws IOException {
        if (!Files.isDirectory(dir)) {
            logger.trace("Directory not exists: {}", dir);
            return null;
        }
        Collection<Path> files =
                Files.walk(dir)
                        .filter(p -> Files.isRegularFile(p) && p.toString().endsWith(".class"))
                        .collect(Collectors.toList());
        for (Path file : files) {
            Path p = dir.relativize(file);
            String className = p.toString();
            className = className.substring(0, className.lastIndexOf('.'));
            className = className.replace(File.separatorChar, '.');
            T implemented = initClass(cl, className);
            if (implemented != null) {
                return implemented;
            }
        }
        return null;
    }

    private static <T> T scanJarFile(ClassLoader cl, Path path) throws IOException {
        try (JarFile jarFile = new JarFile(path.toFile())) {
            Enumeration<JarEntry> en = jarFile.entries();
            while (en.hasMoreElements()) {
                JarEntry entry = en.nextElement();
                String fileName = entry.getName();
                if (fileName.endsWith(".class")) {
                    fileName = fileName.substring(0, fileName.lastIndexOf('.'));
                    fileName = fileName.replace('/', '.');
                    T implemented = initClass(cl, fileName);
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
     * @param className the class to be loaded
     * @param <T> the type of the class
     * @return an instance of the class, null if the class not found
     */
    @SuppressWarnings("unchecked")
    public static <T> T initClass(ClassLoader cl, String className) {
        try {
            Class<?> clazz = Class.forName(className, true, cl);
            Constructor<T> constructor = (Constructor<T>) clazz.getConstructor();
            return constructor.newInstance();
        } catch (Throwable e) {
            logger.trace("Not able to load Object", e);
        }
        return null;
    }
}
