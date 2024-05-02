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
package ai.djl.util;

import ai.djl.util.cuda.CudaUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.Enumeration;
import java.util.Properties;

/**
 * The platform contains information regarding the version, os, and build flavor of the native code.
 */
public final class Platform {

    private static final Logger logger = LoggerFactory.getLogger(Platform.class);

    private String version;
    private String apiVersion;
    private String osPrefix;
    private String osArch;
    private String flavor;
    private String cudaArch;
    private String[] libraries;
    private boolean placeholder;

    /** Constructor used only for {@link Platform#fromSystem()}. */
    private Platform() {}

    /**
     * Returns the platform that matches current operating system.
     *
     * @param engine the name of the engine
     * @param overrideVersion the version of the engine
     * @return the platform that matches current operating system
     */
    public static Platform detectPlatform(String engine, String overrideVersion) {
        Platform platform = Platform.fromSystem(engine);
        platform.version = overrideVersion;
        return platform;
    }

    /**
     * Returns the platform that matches current operating system.
     *
     * @param engine the name of the engine
     * @return the platform that matches current operating system
     */
    public static Platform detectPlatform(String engine) {
        String nativeProp = "native/lib/" + engine + ".properties";
        Enumeration<URL> urls;
        try {
            urls = ClassLoaderUtils.getContextClassLoader().getResources(nativeProp);
        } catch (IOException e) {
            throw new AssertionError("Failed to list property files.", e);
        }

        Platform systemPlatform = Platform.fromSystem(engine);
        Platform placeholder = null;
        while (urls.hasMoreElements()) {
            URL url = urls.nextElement();
            Platform platform = Platform.fromUrl(url);
            platform.apiVersion = systemPlatform.apiVersion;
            if (platform.isPlaceholder()) {
                placeholder = platform;
            } else if (platform.matches(systemPlatform)) {
                logger.info("Found matching platform from: {}", url);
                return platform;
            } else {
                logger.info("Ignore mismatching platform from: {}", url);
            }
        }
        if (placeholder != null) {
            logger.info("Found placeholder platform from: {}", placeholder);
            return placeholder;
        }

        if (systemPlatform.version == null) {
            throw new AssertionError("No " + engine + " version found in property file.");
        }
        if (systemPlatform.apiVersion == null) {
            throw new AssertionError("No " + engine + " djl_version found in property file.");
        }
        return systemPlatform;
    }

    /**
     * Returns the platform that parsed from "engine".properties file.
     *
     * @param url the url to the "engine".properties file
     * @return the platform that parsed from "engine".properties file
     */
    static Platform fromUrl(URL url) {
        Platform platform = Platform.fromSystem();
        try (InputStream conf = url.openStream()) {
            Properties prop = new Properties();
            prop.load(conf);
            platform.version = prop.getProperty("version");
            if (platform.version == null) {
                throw new IllegalArgumentException(
                        "version key is required in <engine>.properties file.");
            }
            platform.placeholder = prop.getProperty("placeholder") != null;
            String flavor = prop.getProperty("flavor");
            if (flavor != null) {
                platform.flavor = flavor;
            }
            String flavorPrefixedClassifier = prop.getProperty("classifier", "");
            String libraryList = prop.getProperty("libraries", "");
            if (libraryList.isEmpty()) {
                platform.libraries = Utils.EMPTY_ARRAY;
            } else {
                platform.libraries = libraryList.split(",");
            }

            if (!flavorPrefixedClassifier.isEmpty()) {
                String[] tokens = flavorPrefixedClassifier.split("-");
                if (flavor != null) {
                    platform.osPrefix = tokens[0];
                    platform.osArch = tokens[1];
                } else {
                    platform.flavor = tokens[0];
                    platform.osPrefix = tokens[1];
                    platform.osArch = tokens[2];
                }
            }
        } catch (IOException e) {
            throw new IllegalStateException("Failed to read property file: " + url, e);
        }
        return platform;
    }

    /**
     * Returns the system platform.
     *
     * @param engine the name of the engine
     * @return the platform representing the system (without an "engine".properties file)
     */
    public static Platform fromSystem(String engine) {
        String engineProp = engine + "-engine.properties";
        String versionKey = engine + "_version";
        Platform platform = fromSystem();
        platform.placeholder = true;

        try {
            URL url = ClassLoaderUtils.getResource(engineProp);
            if (url != null) {
                try (InputStream is = url.openStream()) {
                    Properties prop = new Properties();
                    prop.load(is);
                    platform.version = prop.getProperty(versionKey);
                    platform.apiVersion = prop.getProperty("djl_version");
                }
            }
        } catch (IOException e) {
            throw new AssertionError("Failed to read property file: " + engineProp, e);
        }
        return platform;
    }

    /**
     * Returns the platform for the current system.
     *
     * @return the platform for the current system
     */
    static Platform fromSystem() {
        Platform platform = new Platform();
        String osName = System.getProperty("os.name");
        if (osName.startsWith("Win")) {
            platform.osPrefix = "win";
        } else if (osName.startsWith("Mac")) {
            platform.osPrefix = "osx";
        } else if (osName.startsWith("Linux")) {
            platform.osPrefix = "linux";
        } else {
            throw new AssertionError("Unsupported platform: " + osName);
        }
        platform.osArch = System.getProperty("os.arch");
        if ("amd64".equals(platform.osArch)) {
            platform.osArch = "x86_64";
        }
        if (CudaUtils.getGpuCount() > 0) {
            platform.flavor = "cu" + CudaUtils.getCudaVersionString();
            platform.cudaArch = CudaUtils.getComputeCapability(0);
        } else {
            platform.flavor = "cpu";
        }
        return platform;
    }

    /**
     * Returns the Engine version.
     *
     * @return the Engine version
     */
    public String getVersion() {
        return version;
    }

    /**
     * Returns the Engine API version.
     *
     * @return the Engine API version
     */
    public String getApiVersion() {
        return apiVersion;
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
     * Returns the os architecture (x86_64, aar64, etc).
     *
     * @return the os architecture (x86_64, aar64, etc)
     */
    public String getOsArch() {
        return osArch;
    }

    /**
     * Returns the engine build flavor.
     *
     * @return the engine build flavor
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
        return osPrefix + '-' + osArch;
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
     * @param system the platform to compare it to
     * @return true if the platforms match
     */
    public boolean matches(Platform system) {
        if (!osPrefix.equals(system.osPrefix) || !osArch.equals(system.osArch)) {
            return false;
        }
        if (flavor.startsWith("cpu") || "mkl".equals(flavor)) {
            // CPU package can run on all system platform
            return true;
        }

        // native package can run on system which CUDA version is greater or equal
        if (system.flavor.startsWith("cu")
                && Integer.parseInt(flavor.substring(2, 5))
                        <= Integer.parseInt(system.flavor.substring(2, 5))) {
            return true;
        }
        logger.warn("The bundled library: {}} doesn't match system: {}", this, system);
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return flavor + '-' + getClassifier() + ':' + version;
    }
}
