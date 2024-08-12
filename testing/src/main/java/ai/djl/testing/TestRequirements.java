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
package ai.djl.testing;

import ai.djl.engine.Engine;
import ai.djl.util.Utils;

import org.testng.SkipException;

import java.util.Calendar;

/**
 * This utility class is used to define test requirements.
 *
 * <p>When the test requirements are not fulfilled, the test is skipped with a {@link
 * SkipException}.
 */
@SuppressWarnings("PMD.TestClassWithoutTestCases")
public final class TestRequirements {

    private TestRequirements() {}

    /** Requires a test runs as part of the nightly suite, but not standard local or CI builds. */
    public static void nightly() {
        if (!Boolean.getBoolean("nightly")) {
            throw new SkipException("This test requires the nightly flag to run");
        }
    }

    /** Requires a test only run one day per week (Saturday). */
    public static void weekly() {
        if (Calendar.SATURDAY != Calendar.getInstance().get(Calendar.DAY_OF_WEEK)) {
            throw new SkipException("This test only runs one day per week (Saturday)");
        }
    }

    /** Requires a test not be run in offline mode. */
    public static void notOffline() {
        if (Utils.isOfflineMode()) {
            throw new SkipException("This test can not run while offline");
        }
    }

    /**
     * Requires a test have at least one gpu.
     *
     * @param engineName the engine name
     */
    public static void gpu(String engineName) {
        if (Engine.getEngine(engineName).getGpuCount() == 0) {
            throw new SkipException("This test requires a GPU to run");
        }
    }

    /**
     * Requires the test runs on non-GPU.
     *
     * @param engineName the engine name
     */
    public static void notGpu(String engineName) {
        if (Engine.getEngine(engineName).getGpuCount() > 0) {
            throw new SkipException("This test requires non-GPU to run");
        }
    }

    /** Requires that the test runs on macOS M1. */
    public static void macosM1() {
        if (!System.getProperty("os.name").toLowerCase().startsWith("mac")
                || !"aarch64".equals(System.getProperty("os.arch"))) {
            throw new SkipException("This test requires a macOS M1.");
        }
    }

    /** Requires that the test runs on macOS M1, Linux or Windows, not mac x86_64. */
    public static void notMacX86() {
        if (System.getProperty("os.name").toLowerCase().startsWith("mac")
                && "x86_64".equals(System.getProperty("os.arch"))) {
            throw new SkipException("macOS x86_64 is not supported.");
        }
    }

    /** Requires that the test runs on OSX or linux, not windows. */
    public static void notWindows() {
        if (System.getProperty("os.name").toLowerCase().startsWith("win")) {
            throw new SkipException("This test requires a non-windows os.");
        }
    }

    /** Requires that the test runs on x86_64 arch. */
    public static void notArm() {
        if ("aarch64".equals(System.getProperty("os.arch"))) {
            throw new SkipException("This test requires a non-arm os.");
        }
    }
}
