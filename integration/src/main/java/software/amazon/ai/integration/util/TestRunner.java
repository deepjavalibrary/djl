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
package software.amazon.ai.integration.util;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import software.amazon.ai.Context;
import software.amazon.ai.engine.Engine;

public final class TestRunner {
    private static final Logger logger = LogUtils.getLogger(TestRunner.class);
    private static final Map<String, Class<?>> TEST_CLASSES =
            getClasses("software.amazon.ai.integration.tests");

    private TestRunner() {}

    public static void main(String[] args)
            throws ParseException, InstantiationException, NoSuchMethodException,
                    IllegalAccessException, InvocationTargetException {
        DefaultParser parser = new DefaultParser();
        CommandLine cmd = parser.parse(Arguments.getOptions(), args, null, false);
        Arguments arguments = new Arguments(cmd);

        Duration duration = Duration.ofMinutes(arguments.getDuration());
        int iteration = arguments.getIteration();
        logger.info(
                "Running integration tests on: {}, iteration: {}, duration: {} minutes.",
                Context.defaultContext(),
                iteration,
                duration.toMinutes());
        long init = System.nanoTime();
        String version = Engine.getInstance().getVersion();
        long loaded = System.nanoTime();
        logger.info(
                String.format("Load library %s in %.3f ms.", version, (loaded - init) / 1000000f));

        while (!duration.isNegative()) {
            // TODO: collect performance data
            long begin = System.currentTimeMillis();
            int[] result =
                    runTests(
                            arguments.getPackageName(),
                            arguments.getClassName(),
                            arguments.getMethodName());
            if (result[0] != 0 && result[1] == 0) {
                logger.info("All tests passed");
            } else {
                logger.error("{} of {} tests failed", result[1], result[0]);
            }
            long delta = System.currentTimeMillis() - begin;
            duration = duration.minus(Duration.ofMillis(delta));
        }
    }

    private static int[] runTests(String packageName, String className, String methodName)
            throws InstantiationException, NoSuchMethodException, IllegalAccessException,
                    InvocationTargetException {
        Map<String, Class<?>> classMap = TEST_CLASSES;
        if (packageName != null) {
            classMap = getClasses(packageName);
        }
        if (className == null && methodName == null) {
            return runAllTestsIn(classMap);
        }
        if (className == null) {
            return runTest(classMap, methodName);
        }

        if (methodName == null) {
            if (classMap.get(className) == null) {
                throw new NoSuchMethodException();
            }
            return runAllTestsIn(classMap.get(className));
        }
        return null;
    }

    private static int[] runAllTests()
            throws InstantiationException, IllegalAccessException, InvocationTargetException {
        int failure = 0;
        int total = 0;
        for (Map.Entry<String, Class<?>> entry : TEST_CLASSES.entrySet()) {
            int[] result = runAllTestsIn(entry.getValue());
            total += result[0];
            failure += result[1];
        }
        return new int[] {total, failure};
    }

    private static int[] runAllTestsIn(Map<String, Class<?>> classes)
            throws InstantiationException, IllegalAccessException, InvocationTargetException {
        int failure = 0;
        int total = 0;
        for (Map.Entry<String, Class<?>> entry : classes.entrySet()) {
            int[] result = runAllTestsIn(entry.getValue());
            total += result[0];
            failure += result[1];
        }
        return new int[] {total, failure};
    }

    private static int[] runAllTestsIn(Class<?> klass)
            throws InstantiationException, IllegalAccessException, InvocationTargetException {
        int failure = 0;
        int total = 0;
        Method[] methods = klass.getMethods();
        for (Method method : methods) {
            try {
                Constructor<?> ctor = klass.getConstructor();
                int[] result = runTest(klass, method, ctor.newInstance());
                total += result[0];
                failure += result[1];
            } catch (NoSuchMethodException ignore) {
            }
        }
        return new int[] {total, failure};
    }

    private static int[] runTest(Map<String, Class<?>> classes, String methodName)
            throws InstantiationException, NoSuchMethodException, IllegalAccessException,
                    InvocationTargetException {
        int failure = 0;
        int total = 0;
        for (Map.Entry<String, Class<?>> entry : classes.entrySet()) {
            Class<?> klass = entry.getValue();
            Method method = klass.getMethod(methodName);
            if (method != null) {
                int[] result = runTest(klass, method);
                total += result[0];
                failure += result[1];
            }
        }
        return new int[] {total, failure};
    }

    private static int[] runTest(Class<?> klass, Method method)
            throws InstantiationException, NoSuchMethodException, IllegalAccessException,
                    InvocationTargetException {
        Constructor<?> ctor = klass.getConstructor();
        return runTest(klass, method, ctor.newInstance());
    }

    private static int[] runTest(Class<?> klass, Method method, Object instance) {
        try {
            if (method.isAnnotationPresent(RunAsTest.class)) {
                method.invoke(instance);
                logger.info("Test {}.{} PASSED", klass.getName(), method.getName());
                return new int[] {1, 0};
            }
        } catch (IllegalAccessException | InvocationTargetException e) {
            logger.info("Test {}.{} FAILED", klass.getName(), method.getName());
            logger.error("", e.getCause());
            return new int[] {1, 1};
        }
        return new int[] {0, 0};
    }

    /**
     * Scans all classes accessible from the context class loader which belong to the given package
     * and subpackages or jar.
     *
     * @param path The path to the resource
     * @return The classes
     */
    private static Map<String, Class<?>> getClasses(String path) {
        if (path.endsWith("jar")) {
            return getClassesFromjar(path);
        }
        return getClassesFromPackage(path);
    }

    private static Map<String, Class<?>> getClassesFromPackage(String packageName) {
        ClassLoader classLoader =
                Objects.requireNonNull(Thread.currentThread().getContextClassLoader());
        Map<String, Class<?>> classes = new ConcurrentHashMap<>();
        String path = packageName.replace('.', '/');
        List<File> dirs = new ArrayList<>();
        try {
            Enumeration<URL> resources = classLoader.getResources(path);
            while (resources.hasMoreElements()) {
                URL resource = resources.nextElement();
                dirs.add(new File(resource.getFile()));
            }
        } catch (IOException e) {
            logger.error("", e);
        }
        for (File directory : dirs) {
            addAll(classes, findClasses(directory, packageName));
        }
        return classes;
    }

    private static Map<String, Class<?>> getClassesFromjar(String pathToJar) {
        Map<String, Class<?>> classes = new ConcurrentHashMap<>();
        try (JarFile jarFile = new JarFile(pathToJar)) {
            Enumeration<JarEntry> e = jarFile.entries();

            URL[] urls = {new URL("jar:file:" + pathToJar + "!/")};
            URLClassLoader cl = URLClassLoader.newInstance(urls);

            while (e.hasMoreElements()) {
                JarEntry je = e.nextElement();
                if (je.isDirectory() || !je.getName().endsWith(".class")) {
                    continue;
                }
                String className = je.getName().substring(0, je.getName().length() - 6);
                className = className.replace('/', '.');
                Class<?> klass = cl.loadClass(className);
                classes.put(className, klass);
            }
        } catch (IOException | ClassNotFoundException e) {
            logger.error("IOException: e=", e.getCause());
        }
        return classes;
    }

    /**
     * Recursive method used to find all classes in a given directory and subdirs.
     *
     * @param directory The base directory
     * @param packageName The package name for classes found inside the base directory
     * @return The classes
     */
    private static Map<String, Class<?>> findClasses(File directory, String packageName) {
        Map<String, Class<?>> classes = new ConcurrentHashMap<>();
        if (!directory.exists()) {
            return classes;
        }
        File[] files = directory.listFiles();
        if (files == null) {
            return classes;
        }
        for (File file : files) {
            if (file.isDirectory()) {
                assert !file.getName().contains(".");
                addAll(classes, findClasses(file, packageName + "." + file.getName()));
            } else if (file.getName().endsWith(".class")) {
                try {
                    classes.put(
                            file.getName(),
                            Class.forName(
                                    packageName
                                            + '.'
                                            + file.getName()
                                                    .substring(0, file.getName().length() - 6)));
                } catch (ClassNotFoundException e) {
                    logger.error("ClassNotFound: fileName={}", file.getName());
                }
            }
        }
        return classes;
    }

    private static void addAll(Map<String, Class<?>> first, Map<String, Class<?>> second) {
        for (Map.Entry<String, Class<?>> entry : second.entrySet()) {
            first.put(entry.getKey(), entry.getValue());
        }
    }
}
