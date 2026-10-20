/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

public class ClassLoaderUtilsTest {

    private static final String PROPERTY = "ai.djl.allowed_model_classes";

    /** Set by the static initializer of each nested class below, and by nothing else. */
    static String initialized;

    private String savedAllowlist;

    @BeforeMethod
    public void saveAllowlist() {
        // The environment variable takes precedence over the system property, so an environment
        // that sets it would make every property below a no-op.
        if (Utils.getenv("DJL_ALLOWED_MODEL_CLASSES") != null) {
            throw new SkipException("DJL_ALLOWED_MODEL_CLASSES is set in the environment");
        }
        // The build forwards every ai.djl.* system property into the test JVM, so this may already
        // be set by the developer running the suite. Record it, clear it so the assertions below
        // see the default, and put it back in cleanup().
        savedAllowlist = System.getProperty(PROPERTY);
        System.clearProperty(PROPERTY);
    }

    @AfterMethod
    public static void clearCanary() {
        initialized = null;
    }

    @AfterMethod
    public void cleanup() {
        if (savedAllowlist == null) {
            System.clearProperty(PROPERTY);
        } else {
            System.setProperty(PROPERTY, savedAllowlist);
        }
    }

    @Test
    public void testUnsetAllowlistAllowsAnything() {
        Assert.assertTrue(ClassLoaderUtils.isClassAllowed("com.example.Anything"));

        // Empty is treated as unset rather than as an allowlist that admits nothing, so clearing
        // the value in a launch script does not break a deployment.
        System.setProperty(PROPERTY, "");
        Assert.assertTrue(ClassLoaderUtils.isClassAllowed("com.example.Anything"));
    }

    @Test
    public void testEntriesMatchExactlyOrByPackage() {
        System.setProperty(PROPERTY, "com.example.Exact,com.other.allowed.*");

        Assert.assertTrue(ClassLoaderUtils.isClassAllowed("com.example.Exact"));
        Assert.assertTrue(ClassLoaderUtils.isClassAllowed("com.other.allowed.Translator"));
        // A .* entry covers subpackages too.
        Assert.assertTrue(ClassLoaderUtils.isClassAllowed("com.other.allowed.nested.Translator"));
        // A nested class of an allowed class is a distinct name, and is covered by the package
        // entry it sits under.
        Assert.assertTrue(ClassLoaderUtils.isClassAllowed("com.other.allowed.Outer$Inner"));

        // An exact entry is not a prefix.
        Assert.assertFalse(ClassLoaderUtils.isClassAllowed("com.example.ExactlyNot"));
        Assert.assertFalse(ClassLoaderUtils.isClassAllowed("com.example.Exact.Nested"));
        Assert.assertFalse(ClassLoaderUtils.isClassAllowed("com.other.Allowed"));
        // A package whose name merely starts with an allowed one is excluded: the retained dot
        // makes the comparison end at a package boundary.
        Assert.assertFalse(ClassLoaderUtils.isClassAllowed("com.other.allowedevil.Translator"));
    }

    @Test
    public void testEntriesAreTrimmed() {
        System.setProperty(PROPERTY, " com.example.Exact , com.other.allowed.* ");
        Assert.assertTrue(ClassLoaderUtils.isClassAllowed("com.example.Exact"));
        Assert.assertTrue(ClassLoaderUtils.isClassAllowed("com.other.allowed.Translator"));
    }

    @Test
    public void testIncompatibleClassIsNotInitialized() {
        ClassLoader cl = ClassLoaderUtils.getContextClassLoader();
        String incompatible = ClassLoaderUtilsTest.class.getName() + "$NotARunnable";

        // Named by class name only: resolving it must not run its initializer, which is what
        // initialize=false buys. Flipping that back to true fails here.
        Assert.assertNull(ClassLoaderUtils.initClass(cl, Runnable.class, incompatible));
        Assert.assertNull(initialized);

        // Control: a compatible class does load, instantiate and initialize.
        String compatible = ClassLoaderUtilsTest.class.getName() + "$ARunnable";
        Assert.assertNotNull(ClassLoaderUtils.initClass(cl, Runnable.class, compatible));
        Assert.assertEquals(initialized, "ARunnable");
    }

    /** Has a public no-arg constructor but is not a {@link Runnable}. */
    public static final class NotARunnable {

        static {
            initialized = "NotARunnable";
        }

        public NotARunnable() {}
    }

    /** The control: same shape, but implements the requested type. */
    public static final class ARunnable implements Runnable {

        static {
            initialized = "ARunnable";
        }

        public ARunnable() {}

        /** {@inheritDoc} */
        @Override
        public void run() {}
    }
}
