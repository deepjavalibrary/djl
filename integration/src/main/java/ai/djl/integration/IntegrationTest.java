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
package ai.djl.integration;

import ai.djl.integration.util.Arguments;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.stream.Collectors;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.AfterClass;
import org.testng.annotations.AfterTest;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

public class IntegrationTest {

    private static final Logger logger = LoggerFactory.getLogger(IntegrationTest.class);

    private static final String PACKAGE_NAME = "ai.djl.integration.tests.";

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
            List<TestClass> tests = listTests(arguments);

            while (!duration.isNegative()) {
                long begin = System.currentTimeMillis();

                runTests(tests);

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

    private void runTests(List<TestClass> tests) {
        long totalMethodCount = 0;
        for (TestClass testClass : tests) {
            logger.info("Running test {} ...", testClass.getName());
            int testCount = testClass.getTestCount();
            totalMethodCount += testCount;

            try {
                if (!testClass.beforeClass()) {
                    totalFailed += totalMethodCount;
                    continue;
                }

                for (int i = 0; i < testCount; ++i) {
                    if (!testClass.runTest(i)) {
                        ++totalFailed;
                    }
                }
            } finally {
                testClass.afterClass();
            }
        }
        if (totalFailed > 0) {
            logger.error("Failed {} out of {} tests", totalFailed, totalMethodCount);
        } else {
            logger.info("Passed all {} tests", totalMethodCount);
        }
    }

    private static List<TestClass> listTests(Arguments arguments) {
        String className = arguments.getClassName();
        String methodName = arguments.getMethodName();
        List<TestClass> tests = new ArrayList<>();
        try {
            if (className != null) {
                Class<?> clazz;
                if (className.startsWith(PACKAGE_NAME)) {
                    clazz = Class.forName(className);
                } else {
                    clazz = Class.forName(PACKAGE_NAME + className);
                }
                tests.add(getTestsInClass(clazz, methodName));
            } else {
                List<Class<?>> classes = listTestClasses(IntegrationTest.class);
                for (Class<?> clazz : classes) {
                    tests.add(getTestsInClass(clazz, methodName));
                }
            }
        } catch (ReflectiveOperationException | IOException | URISyntaxException e) {
            logger.error("Failed to resolve test class.", e);
        }
        return tests;
    }

    private static TestClass getTestsInClass(Class<?> clazz, String methodName)
            throws ReflectiveOperationException {
        Constructor<?> ctor = clazz.getConstructor();
        Object obj = ctor.newInstance();
        TestClass testClass = new TestClass(obj);

        for (Method method : clazz.getDeclaredMethods()) {
            Test testMethod = method.getAnnotation(Test.class);
            if (testMethod != null) {
                if (testMethod.enabled()
                        && (methodName == null || methodName.equals(method.getName()))) {
                    testClass.addTestMethod(method);
                }
                continue;
            }
            BeforeClass beforeClass = method.getAnnotation(BeforeClass.class);
            if (beforeClass != null) {
                testClass.addBeforeClass(method);
                continue;
            }
            AfterClass afterClass = method.getAnnotation(AfterClass.class);
            if (afterClass != null) {
                testClass.addAfterClass(method);
                continue;
            }
            BeforeTest beforeTest = method.getAnnotation(BeforeTest.class);
            if (beforeTest != null) {
                testClass.addBeforeTest(method);
                continue;
            }
            AfterTest afterTest = method.getAnnotation(AfterTest.class);
            if (afterTest != null) {
                testClass.addAfterTest(method);
            }
        }

        return testClass;
    }

    private static List<Class<?>> listTestClasses(Class<?> clazz)
            throws IOException, ClassNotFoundException, URISyntaxException {
        URL url = clazz.getProtectionDomain().getCodeSource().getLocation();
        String path = url.getPath();

        if (!"file".equalsIgnoreCase(url.getProtocol())) {
            return Collections.emptyList();
        }

        List<Class<?>> classList = new ArrayList<>();

        Path classPath = Paths.get(url.toURI());
        if (Files.isDirectory(classPath)) {
            Collection<Path> files =
                    Files.walk(classPath)
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
            try (JarFile jarFile = new JarFile(classPath.toFile())) {
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

    private static final class TestClass {

        private Object object;
        private List<Method> testMethods;
        private List<Method> beforeClass;
        private List<Method> afterClass;
        private List<Method> beforeTest;
        private List<Method> afterTest;

        public TestClass(Object object) {
            this.object = object;
            testMethods = new ArrayList<>();
            beforeClass = new ArrayList<>();
            afterClass = new ArrayList<>();
            beforeTest = new ArrayList<>();
            afterTest = new ArrayList<>();
        }

        public void addTestMethod(Method method) {
            testMethods.add(method);
        }

        public void addBeforeClass(Method method) {
            beforeClass.add(method);
        }

        public void addAfterClass(Method method) {
            afterClass.add(method);
        }

        public void addBeforeTest(Method method) {
            beforeTest.add(method);
        }

        public void addAfterTest(Method method) {
            afterTest.add(method);
        }

        public boolean beforeClass() {
            try {
                for (Method method : beforeClass) {
                    method.invoke(object);
                }
                return true;
            } catch (InvocationTargetException | IllegalAccessException e) {
                logger.error("", e.getCause());
            }
            return false;
        }

        public void afterClass() {
            try {
                for (Method method : afterClass) {
                    method.invoke(object);
                }
            } catch (InvocationTargetException | IllegalAccessException e) {
                logger.error("", e.getCause());
            }
        }

        public boolean beforeTest() {
            try {
                for (Method method : beforeTest) {
                    method.invoke(object);
                }
                return true;
            } catch (InvocationTargetException | IllegalAccessException e) {
                logger.error("", e.getCause());
            }
            return false;
        }

        public void afterTest() {
            try {
                for (Method method : afterTest) {
                    method.invoke(object);
                }
            } catch (InvocationTargetException | IllegalAccessException e) {
                logger.error("", e.getCause());
            }
        }

        public boolean runTest(int index) {
            if (!beforeTest()) {
                return false;
            }

            Method method = testMethods.get(index);
            try {
                method.invoke(object);
                logger.info("Test {}.{} PASSED", getName(), method.getName());
            } catch (IllegalAccessException | InvocationTargetException e) {
                if (notExpected(method, e)) {
                    logger.error("Test {}.{} FAILED", getName(), method.getName());
                    logger.error("", e.getCause());
                    return false;
                }
            } finally {
                afterTest();
            }
            return true;
        }

        public int getTestCount() {
            return testMethods.size();
        }

        public String getName() {
            return object.getClass().getName();
        }

        private static boolean notExpected(Method method, Exception e) {
            Test test = method.getAnnotation(Test.class);
            Class<?>[] exceptions = test.expectedExceptions();
            if (exceptions.length > 0) {
                Throwable exception = e.getCause();
                for (Class<?> c : exceptions) {
                    if (c.isInstance(exception)) {
                        return false;
                    }
                }
            }
            return true;
        }
    }
}
