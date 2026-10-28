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
import org.testng.annotations.AfterMethod;
import org.testng.annotations.Test;

public class ClassLoaderUtilsTest {

    /** Set by the static initializer of each nested class below, and by nothing else. */
    static String initialized;

    @AfterMethod
    public static void clearCanary() {
        initialized = null;
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
