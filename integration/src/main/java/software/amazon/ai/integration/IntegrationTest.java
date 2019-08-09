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
package software.amazon.ai.integration;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Enumeration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.stream.Collectors;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import software.amazon.ai.integration.util.Arguments;
import software.amazon.ai.integration.util.LogUtils;
import software.amazon.ai.integration.util.RunAsTest;

public class IntegrationTest {

    private static final Logger logger = LogUtils.getLogger(IntegrationTest.class);

    private static final String PACKAGE_NAME = "software.amazon.ai.integration.tests.";

    private int totalFailed;

    public static void main(String[] args) {
        new IntegrationTest().runTests(args);
    }

    public boolean runTests(String[] args) {
        Options options = Arguments.getOptions();
        try {
            DefaultParser parser = new DefaultParser();
            CommandLine cmd = parser.parse(options, args, null, false);
            Arguments arguments = new Arguments(cmd);

            Duration duration = Duration.ofMinutes(arguments.getDuration());
            int iteration = arguments.getIteration();

            Map<Object, List<Method>> tests = listTests(arguments);

            while (!duration.isNegative()) {
                long begin = System.currentTimeMillis();

                runTests(tests, iteration);

                long delta = System.currentTimeMillis() - begin;
                duration = duration.minus(Duration.ofMillis(delta));
            }
        } catch (ParseException e) {
            HelpFormatter formatter = new HelpFormatter();
            formatter.setLeftPadding(1);
            formatter.setWidth(120);
            formatter.printHelp(e.getMessage(), options);
        } catch (Throwable t) {
            logger.error("Unexpected error", t);
        }
        return totalFailed == 0;
    }

    private void runTests(Map<Object, List<Method>> tests, int iteration) {
        long totalMethodCount = 0;
        for (Map.Entry<Object, List<Method>> entry : tests.entrySet()) {
            Object testObject = entry.getKey();
            String testClass = testObject.getClass().getName();
            List<Method> methods = entry.getValue();

            logger.info("Running test {}, iteration: {}", testClass, iteration);

            int failed = 0;
            for (Method method : methods) {
                // TODO: collect performance data
                for (int i = 0; i < iteration; i++) {
                    try {
                        if (method.isAnnotationPresent(RunAsTest.class)) {
                            totalMethodCount++;
                            method.invoke(testObject);
                            logger.info("Test {}.{} PASSED", testClass, method.getName());
                        }
                    } catch (IllegalAccessException | InvocationTargetException e) {
                        logger.error("Test {}.{} FAILED", testClass, method.getName());
                        logger.error("", e.getCause());
                        ++failed;
                        break;
                    }
                }
            }
            totalFailed += failed;
        }
        if (totalFailed > 0) {
            logger.error("Failed {} out of {} tests", totalFailed, totalMethodCount);
        } else {
            logger.info("Passed all {} tests", totalMethodCount);
        }
    }

    private static Map<Object, List<Method>> listTests(Arguments arguments) {
        String className = arguments.getClassName();
        String methodName = arguments.getMethodName();
        Map<Object, List<Method>> map = new LinkedHashMap<>(); // NOPMD
        try {
            if (className != null) {
                if (!className.startsWith(PACKAGE_NAME)) {
                    className = PACKAGE_NAME + className; // NOPMD
                }
                Class<?> clazz = Class.forName(className);
                Constructor<?> ctor = clazz.getConstructor();
                Object obj = ctor.newInstance();
                if (methodName != null) {
                    Method method = clazz.getDeclaredMethod(methodName);
                    if (method.isAnnotationPresent(RunAsTest.class)) {
                        map.put(obj, Collections.singletonList(method));
                    } else {
                        logger.error("Invalid method name: {}. Not a test.", methodName);
                    }
                } else {
                    map.put(obj, getTestsInClass(clazz));
                }
            } else {
                List<Class<?>> classes = listTestClasses(IntegrationTest.class);
                for (Class<?> clazz : classes) {
                    Constructor<?> ctor = clazz.getConstructor();
                    Object obj = ctor.newInstance();
                    map.put(obj, getTestsInClass(clazz));
                }
            }
        } catch (ReflectiveOperationException | IOException e) {
            logger.error("Failed to resolve test class.", e);
        }
        return map;
    }

    private static List<Method> getTestsInClass(Class<?> clazz) {
        Method[] methods = clazz.getDeclaredMethods();
        return Arrays.stream(methods)
                .filter(o -> o.isAnnotationPresent(RunAsTest.class))
                .collect(Collectors.toList());
    }

    private static List<Class<?>> listTestClasses(Class<?> clazz)
            throws IOException, ClassNotFoundException {
        URL url = clazz.getProtectionDomain().getCodeSource().getLocation();
        String path = url.getPath();

        if (!"file".equalsIgnoreCase(url.getProtocol())) {
            return Collections.emptyList();
        }

        List<Class<?>> classList = new ArrayList<>();

        Path classPath = Paths.get(path);
        if (Files.isDirectory(classPath)) {
            Collection<Path> files =
                    Files.walk(Paths.get(path))
                            .filter(p -> Files.isRegularFile(p) && p.toString().endsWith(".class"))
                            .collect(Collectors.toList());
            for (Path file : files) {
                Path p = classPath.relativize(file);
                String className = p.toString();
                className = className.substring(0, className.lastIndexOf('.'));
                className = className.replace(File.separatorChar, '.');
                if (className.startsWith(PACKAGE_NAME) && !className.contains("$")) {
                    try {
                        classList.add(Class.forName(className));
                    } catch (ExceptionInInitializerError ignore) {
                        // ignore
                    }
                }
            }
        } else if (path.toLowerCase().endsWith(".jar")) {
            try (JarFile jarFile = new JarFile(path)) {
                Enumeration<JarEntry> en = jarFile.entries();
                while (en.hasMoreElements()) {
                    JarEntry entry = en.nextElement();
                    String fileName = entry.getName();
                    if (fileName.endsWith(".class")) {
                        fileName = fileName.substring(0, fileName.lastIndexOf('.'));
                        fileName = fileName.replace('/', '.');
                        if (fileName.startsWith(PACKAGE_NAME)) {
                            try {
                                classList.add(Class.forName(fileName));
                            } catch (ExceptionInInitializerError ignore) {
                                // ignore
                            }
                        }
                    }
                }
            }
        }

        return classList;
    }
}
