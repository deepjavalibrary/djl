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
package ai.djl.util.cuda;

import ai.djl.Device;
import ai.djl.engine.EngineException;
import ai.djl.util.Utils;

import com.sun.jna.Native;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.lang.management.MemoryUsage;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

/** A class containing CUDA utility methods. */
public final class CudaUtils {

    private static final Logger logger = LoggerFactory.getLogger(CudaUtils.class);

    private static final CudaLibrary LIB = loadLibrary();

    private static String[] gpuInfo;

    private CudaUtils() {}

    /**
     * Gets whether CUDA runtime library is in the system.
     *
     * @return {@code true} if CUDA runtime library is in the system
     */
    public static boolean hasCuda() {
        return getGpuCount() > 0;
    }

    /**
     * Returns the number of GPUs available in the system.
     *
     * @return the number of GPUs available in the system
     */
    @SuppressWarnings("PMD.NonThreadSafeSingleton")
    public static int getGpuCount() {
        if (Boolean.getBoolean("ai.djl.util.cuda.fork")) {
            if (gpuInfo == null) {
                gpuInfo = execute(-1); // NOPMD
            }
            try {
                return Integer.parseInt(gpuInfo[0]);
            } catch (NumberFormatException e) {
                return 0;
            }
        }

        if (LIB == null) {
            return 0;
        }
        int[] count = new int[1];
        int result = LIB.cudaGetDeviceCount(count);
        switch (result) {
            case 0:
                return count[0];
            case CudaLibrary.ERROR_NO_DEVICE:
                logger.debug(
                        "No GPU device found: {} ({})", LIB.cudaGetErrorString(result), result);
                return 0;
            case CudaLibrary.INITIALIZATION_ERROR:
            case CudaLibrary.INSUFFICIENT_DRIVER:
            case CudaLibrary.ERROR_NOT_PERMITTED:
            default:
                logger.warn(
                        "Failed to detect GPU count: {} ({})",
                        LIB.cudaGetErrorString(result),
                        result);
                return 0;
        }
    }

    /**
     * Returns the version of CUDA runtime.
     *
     * @return the version of CUDA runtime
     */
    @SuppressWarnings("PMD.NonThreadSafeSingleton")
    public static int getCudaVersion() {
        if (Boolean.getBoolean("ai.djl.util.cuda.fork")) {
            if (gpuInfo == null) {
                gpuInfo = execute(-1);
            }
            int version = Integer.parseInt(gpuInfo[1]);
            if (version == -1) {
                throw new IllegalArgumentException("No cuda device found.");
            }
            return version;
        }

        if (LIB == null) {
            throw new IllegalStateException("No cuda library is loaded.");
        }
        int[] version = new int[1];
        int result = LIB.cudaRuntimeGetVersion(version);
        checkCall(result);
        return version[0];
    }

    /**
     * Returns the version string of CUDA runtime.
     *
     * @return the version string of CUDA runtime
     */
    public static String getCudaVersionString() {
        int version = getCudaVersion();
        int major = version / 1000;
        int minor = (version / 10) % 10;
        return String.format(Locale.ROOT, "%02d", major) + minor;
    }

    /**
     * Returns the CUDA compute capability.
     *
     * @param device the GPU {@link Device} to retrieve
     * @return the CUDA compute capability
     */
    public static String getComputeCapability(int device) {
        if (Boolean.getBoolean("ai.djl.util.cuda.fork")) {
            if (gpuInfo == null) { // NOPMD
                gpuInfo = execute(-1);
            }
            if (device >= gpuInfo.length - 2) {
                throw new IllegalArgumentException("Invalid device: " + device);
            }
            return gpuInfo[device + 2];
        }

        if (LIB == null) {
            throw new IllegalStateException("No cuda library is loaded.");
        }
        int attrComputeCapabilityMajor = 75;
        int attrComputeCapabilityMinor = 76;

        int[] major = new int[1];
        int[] minor = new int[1];
        checkCall(LIB.cudaDeviceGetAttribute(major, attrComputeCapabilityMajor, device));
        checkCall(LIB.cudaDeviceGetAttribute(minor, attrComputeCapabilityMinor, device));

        return String.valueOf(major[0]) + minor[0];
    }

    /**
     * Returns the {@link MemoryUsage} of the specified GPU device.
     *
     * @param device the GPU {@link Device} to retrieve
     * @return the {@link MemoryUsage} of the specified GPU device
     * @throws IllegalArgumentException if {@link Device} is not GPU device or does not exist
     */
    public static MemoryUsage getGpuMemory(Device device) {
        if (!device.isGpu()) {
            throw new IllegalArgumentException("Only GPU device is allowed.");
        }

        if (Boolean.getBoolean("ai.djl.util.cuda.fork")) {
            String[] ret = execute(device.getDeviceId());
            if (ret.length != 3) {
                throw new IllegalArgumentException(ret[0]);
            }
            long total = Long.parseLong(ret[1]);
            long used = Long.parseLong(ret[2]);
            return new MemoryUsage(-1, used, used, total);
        }

        if (LIB == null) {
            throw new IllegalStateException("No GPU device detected.");
        }

        int[] currentDevice = new int[1];
        checkCall(LIB.cudaGetDevice(currentDevice));
        checkCall(LIB.cudaSetDevice(device.getDeviceId()));

        long[] free = new long[1];
        long[] total = new long[1];

        checkCall(LIB.cudaMemGetInfo(free, total));
        checkCall(LIB.cudaSetDevice(currentDevice[0]));

        long committed = total[0] - free[0];
        return new MemoryUsage(-1, committed, committed, total[0]);
    }

    /**
     * The main entrypoint to get CUDA information with command line.
     *
     * @param args the command line arguments.
     */
    @SuppressWarnings("PMD.SystemPrintln")
    public static void main(String[] args) {
        int gpuCount = getGpuCount();
        if (args.length == 0) {
            if (gpuCount <= 0) {
                System.out.println("0,-1");
                return;
            }
            int cudaVersion = getCudaVersion();
            StringBuilder sb = new StringBuilder();
            sb.append(gpuCount).append(',').append(cudaVersion);
            for (int i = 0; i < gpuCount; ++i) {
                sb.append(',').append(getComputeCapability(i));
            }
            System.out.println(sb);
            return;
        }
        try {
            int deviceId = Integer.parseInt(args[0]);
            if (deviceId < 0 || deviceId >= gpuCount) {
                System.out.println("Invalid device: " + deviceId);
                return;
            }
            MemoryUsage mem = getGpuMemory(Device.gpu(deviceId));
            String cc = getComputeCapability(deviceId);
            System.out.println(cc + ',' + mem.getMax() + ',' + mem.getUsed());
        } catch (NumberFormatException e) {
            System.out.println("Invalid device: " + args[0]);
        }
    }

    private static CudaLibrary loadLibrary() {
        try {
            if (Boolean.getBoolean("ai.djl.util.cuda.fork")) {
                return null;
            }
            if (System.getProperty("os.name").startsWith("Win")) {
                String path = Utils.getenv("PATH");
                if (path == null) {
                    return null;
                }
                Pattern p = Pattern.compile("cudart64_\\d+\\.dll");
                String cudaPath = Utils.getenv("CUDA_PATH");
                String[] searchPath;
                if (cudaPath == null) {
                    searchPath = path.split(";");
                } else {
                    searchPath = (cudaPath + "\\bin\\;" + path).split(";");
                }
                for (String item : searchPath) {
                    File dir = new File(item);
                    File[] files = dir.listFiles(n -> p.matcher(n.getName()).matches());
                    if (files != null && files.length > 0) {
                        String fileName = files[0].getName();
                        String cudaRt = fileName.substring(0, fileName.length() - 4);
                        logger.debug("Found cudart: {}", files[0].getAbsolutePath());
                        return Native.load(cudaRt, CudaLibrary.class);
                    }
                }
                logger.debug("No cudart library found in path.");
                return null;
            }
            return Native.load("cudart", CudaLibrary.class);
        } catch (UnsatisfiedLinkError e) {
            logger.debug("cudart library not found.");
            logger.trace("", e);
        } catch (LinkageError e) {
            logger.warn("You have a conflict version of JNA in the classpath.");
            logger.debug("", e);
        } catch (SecurityException e) {
            logger.warn("Access denied during loading cudart library.");
            logger.trace("", e);
        }
        return null;
    }

    private static String[] execute(int deviceId) {
        try {
            String javaHome = System.getProperty("java.home");
            String classPath = System.getProperty("java.class.path");
            String os = System.getProperty("os.name");
            List<String> cmd = new ArrayList<>(4);
            if (os.startsWith("Win")) {
                cmd.add(javaHome + "\\bin\\java.exe");
            } else {
                cmd.add(javaHome + "/bin/java");
            }
            cmd.add("-cp");
            cmd.add(classPath);
            cmd.add("ai.djl.util.cuda.CudaUtils");
            if (deviceId >= 0) {
                cmd.add(String.valueOf(deviceId));
            }
            Process ps = new ProcessBuilder(cmd).redirectErrorStream(true).start();
            try (InputStream is = ps.getInputStream()) {
                String line = Utils.toString(is).trim();
                return line.split(",");
            }
        } catch (IOException e) {
            throw new IllegalArgumentException("Failed get GPU information", e);
        }
    }

    private static void checkCall(int ret) {
        if (LIB == null) {
            throw new IllegalStateException("No cuda library is loaded.");
        }
        if (ret != 0) {
            throw new EngineException(
                    "CUDA API call failed: " + LIB.cudaGetErrorString(ret) + " (" + ret + ')');
        }
    }
}
