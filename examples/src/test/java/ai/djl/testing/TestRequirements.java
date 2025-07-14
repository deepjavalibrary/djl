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
import ai.djl.engine.EngineException;
import ai.djl.util.cuda.CudaUtils;

import org.testng.SkipException;

import java.util.Arrays;
import java.util.Calendar;

/**
 * This utility class is used to define test requirements.
 *
 * <p>When the test requirements are not fulfilled, the test is skipped with a {@link
 * SkipException}.
 */
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

    /**
     * Requires a test only with the allowed engine(s).
     *
     * @param engines the engine(s) to run the test with
     */
    public static void engine(String... engines) {
        String engineName = Engine.getDefaultEngineName();
        for (String e : engines) {
            if (engineName.equals(e)) {
                try {
                    Engine.getEngine(engineName);
                } catch (EngineException ex) {
                    throw new SkipException("Engine cannot be loaded.", ex);
                }
                return;
            }
        }
        throw new SkipException(
                "This test requires one of the engines: " + Arrays.toString(engines));
    }

    /**
     * Requires a test have runs on Linux.
     *
     * <p>Avoid running multiple engines on Windows and PyTorch on macos x86_64 machine
     */
    public static void linux() {
        if (!System.getProperty("os.name").toLowerCase().startsWith("linux")) {
            throw new SkipException("This test requires a Linux os.");
        }
    }

    /** Requires a test have at least one gpu. */
    public static void gpu(String engine, int numGpu) {
        if (Engine.getEngine(engine).getGpuCount() < numGpu) {
            throw new SkipException("This test requires " + numGpu + " GPUs to run");
        }
    }

    /** Avoid OOM on GPUs with multiple engines. */
    public static void notGpu() {
        if (CudaUtils.getGpuCount() > 0) {
            throw new SkipException("This test requires CPU only machine to run");
        }
    }

    /** Requires that the test runs on x86_64 arch. */
    public static void notArm() {
        if ("aarch64".equals(System.getProperty("os.arch"))) {
            throw new SkipException("This test requires a non-arm os.");
        }
    }
}
