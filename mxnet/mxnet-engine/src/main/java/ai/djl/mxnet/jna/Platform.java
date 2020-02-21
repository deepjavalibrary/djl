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
package ai.djl.mxnet.jna;

import ai.djl.util.cuda.CudaUtils;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.Properties;

/**
 * The platform contains information regarding the version, os, and build flavor of the MXNet native
 * code.
 */
public class Platform {

    private static final String DEFAULT_VERSION = "1.6.0";

    private String version;
    private String osPrefix;
    private String flavor;
    private String cudaArch;
    private String[] libraries;
    private boolean placeholder;

    /** Constructor used only for {@link Platform#fromSystem()}. */
    Platform() {
        String osName = System.getProperty("os.name");
        if (osName.startsWith("Win")) {
            osPrefix = "win";
        } else if (osName.startsWith("Mac")) {
            osPrefix = "osx";
        } else if (osName.startsWith("Linux")) {
            osPrefix = "linux";
        } else {
            throw new AssertionError("Unsupported platform: " + osName);
        }
        if (CudaUtils.getGpuCount() > 0) {
            flavor = "cu" + CudaUtils.getCudaVersionString() + "mkl";
            cudaArch = CudaUtils.getComputeCapability(0);
        } else {
            flavor = "mkl";
            cudaArch = null;
        }
        version = DEFAULT_VERSION;
    }

    /**
     * Constructor for loading from mxnet.properties files.
     *
     * @param url the url to the mxnet.properties file
     * @throws IOException if the file could not be read
     */
    public Platform(URL url) throws IOException {
        try (InputStream conf = url.openStream()) {
            Properties prop = new Properties();
            prop.load(conf);
            // 1.6.0 later should always has version property
            version = prop.getProperty("version", DEFAULT_VERSION);
            placeholder = prop.getProperty("placeholder") != null;
            String flavorPrefixedClassifier = prop.getProperty("classifier", "");
            libraries = prop.getProperty("libraries", "").split(",");

            if (!"".equals(flavorPrefixedClassifier)) {
                flavor = flavorPrefixedClassifier.split("-")[0];
                osPrefix = flavorPrefixedClassifier.split("-")[1];
            }
        }
    }

    /**
     * Returns the platform for the current system.
     *
     * @return the platform for the current system
     */
    public static Platform fromSystem() {
        return new Platform();
    }

    /**
     * Returns the MXNet version.
     *
     * @return the MXNet version
     */
    public String getVersion() {
        return version;
    }

    /**
     * Returns the os (win, osx, or linux).
     *
     * @return the os (win, osx, or linux)
     */
    public String getOsPrefix() {
        return osPrefix;
    }

    /**
     * Returns the MXNet build flavor.
     *
     * @return the MXNet build flavor
     */
    public String getFlavor() {
        return flavor;
    }

    /**
     * Returns the classifier for the platform.
     *
     * @return the classifier for the platform
     */
    public String getClassifier() {
        return osPrefix + "-x86_64";
    }

    /**
     * Returns the cuda arch.
     *
     * @return the cuda arch
     */
    public String getCudaArch() {
        return cudaArch;
    }

    /**
     * Returns the libraries used in the platform.
     *
     * @return the libraries used in the platform
     */
    public String[] getLibraries() {
        return libraries;
    }

    /**
     * Returns true if the platform is a placeholder.
     *
     * @return true if the platform is a placeholder
     */
    public boolean isPlaceholder() {
        return placeholder;
    }

    /**
     * Returns true the platforms match (os and flavor).
     *
     * @param other the platform to compare it to
     * @return true if the platforms match
     */
    public boolean matches(Platform other) {
        if (osPrefix == null || other.osPrefix == null) {
            return false;
        }
        return osPrefix.equals(other.osPrefix) && flavor.equals(other.flavor);
    }

    /**
     * Returns true the platforms are compatible (same os and only one has cuda).
     *
     * @param other the platform to compare it to
     * @return true if the platforms are compatible
     */
    public boolean compatible(Platform other) {
        if (osPrefix == null || other.osPrefix == null) {
            return false;
        }
        // if both use cuda, the cuda must be the same
        if (flavor.contains("cu") && other.flavor.contains("cu")) {
            return osPrefix.equals(other.osPrefix) && flavor.equals(other.flavor);
        }
        return osPrefix.equals(other.osPrefix);
    }
}
