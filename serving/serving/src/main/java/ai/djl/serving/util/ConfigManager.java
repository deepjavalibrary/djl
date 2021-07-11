/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving.util;

import ai.djl.serving.Arguments;
import ai.djl.util.Utils;
import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.SslContextBuilder;
import io.netty.handler.ssl.util.SelfSignedCertificate;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.GeneralSecurityException;
import java.security.KeyException;
import java.security.KeyFactory;
import java.security.KeyStore;
import java.security.PrivateKey;
import java.security.cert.Certificate;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import java.security.spec.InvalidKeySpecException;
import java.security.spec.PKCS8EncodedKeySpec;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collection;
import java.util.Enumeration;
import java.util.List;
import java.util.Properties;

/** A class that hold configuration information. */
public final class ConfigManager {

    private static final String DEBUG = "debug";
    private static final String INFERENCE_ADDRESS = "inference_address";
    private static final String MANAGEMENT_ADDRESS = "management_address";
    private static final String LOAD_MODELS = "load_models";
    private static final String DEFAULT_WORKERS_PER_MODEL = "default_workers_per_model";
    private static final String NUMBER_OF_NETTY_THREADS = "number_of_netty_threads";
    private static final String JOB_QUEUE_SIZE = "job_queue_size";
    private static final String MAX_IDLE_TIME = "max_idle_time";
    private static final String BATCH_SIZE = "batch_size";
    private static final String MAX_BATCH_DELAY = "max_batch_delay";
    private static final String CORS_ALLOWED_ORIGIN = "cors_allowed_origin";
    private static final String CORS_ALLOWED_METHODS = "cors_allowed_methods";
    private static final String CORS_ALLOWED_HEADERS = "cors_allowed_headers";
    private static final String KEYSTORE = "keystore";
    private static final String KEYSTORE_PASS = "keystore_pass";
    private static final String KEYSTORE_TYPE = "keystore_type";
    private static final String CERTIFICATE_FILE = "certificate_file";
    private static final String PRIVATE_KEY_FILE = "private_key_file";
    private static final String MAX_REQUEST_SIZE = "max_request_size";
    private static final String MODEL_STORE = "model_store";
    private static final String MODEL_URL_PATTERN = "model_url_pattern";
    private static final String PLUGIN_FOLDER = "plugin_folder";

    // Configuration which are not documented or enabled through environment variables
    private static final String USE_NATIVE_IO = "use_native_io";
    private static final String IO_RATIO = "io_ratio";

    private static ConfigManager instance;

    private Properties prop;

    private ConfigManager(Arguments args) {
        prop = new Properties();

        Path file = args.getConfigFile();
        if (file != null) {
            try (InputStream stream = Files.newInputStream(file)) {
                prop.load(stream);
            } catch (IOException e) {
                throw new IllegalArgumentException("Unable to read configuration file", e);
            }
            prop.put("configFile", file.toString());
        }

        String modelStore = args.getModelStore();
        if (modelStore != null) {
            prop.setProperty(MODEL_STORE, modelStore);
        }

        String[] models = args.getModels();
        if (models != null) {
            prop.setProperty(LOAD_MODELS, String.join(",", models));
        }
    }

    /**
     * Initialize the global {@code ConfigManager} instance.
     *
     * @param args the command line arguments
     */
    public static void init(Arguments args) {
        instance = new ConfigManager(args);
    }

    /**
     * Returns the singleton {@code ConfigManager} instance.
     *
     * @return the singleton {@code ConfigManager} instance
     */
    public static ConfigManager getInstance() {
        return instance;
    }

    /**
     * Returns if debug is enabled.
     *
     * @return {@code true} if debug is enabled
     */
    public boolean isDebug() {
        return Boolean.getBoolean("ai.djl.debug")
                || Boolean.parseBoolean(prop.getProperty(DEBUG, "false"));
    }

    /**
     * Returns the models server socket connector.
     *
     * @param type the type of connector
     * @return the {@code Connector}
     */
    public Connector getConnector(Connector.ConnectorType type) {
        String binding;
        if (type == Connector.ConnectorType.MANAGEMENT) {
            binding = prop.getProperty(MANAGEMENT_ADDRESS, "http://127.0.0.1:8080");
        } else {
            binding = prop.getProperty(INFERENCE_ADDRESS, "http://127.0.0.1:8080");
        }
        return Connector.parse(binding, type);
    }

    /**
     * Returns the configured netty threads.
     *
     * @return the configured netty threads
     */
    public int getNettyThreads() {
        return getIntProperty(NUMBER_OF_NETTY_THREADS, 0);
    }

    /**
     * Returns the default job queue size.
     *
     * @return the default job queue size
     */
    public int getJobQueueSize() {
        return getIntProperty(JOB_QUEUE_SIZE, 100);
    }

    /**
     * Returns the default max idle time for workers.
     *
     * @return the default max idle time
     */
    public int getMaxIdleTime() {
        return getIntProperty(MAX_IDLE_TIME, 60);
    }

    /**
     * Returns the default batchSize for workers.
     *
     * @return the default max idle time
     */
    public int getBatchSize() {
        return getIntProperty(BATCH_SIZE, 1);
    }

    /**
     * Returns the default maxBatchDelay for the working queue.
     *
     * @return the default max batch delay
     */
    public int getMaxBatchDelay() {
        return getIntProperty(MAX_BATCH_DELAY, 300);
    }

    /**
     * Returns the default number of workers for a new registered model.
     *
     * @return the default number of workers for a new registered model
     */
    public int getDefaultWorkers() {
        if (isDebug()) {
            return 1;
        }

        int workers = getIntProperty(DEFAULT_WORKERS_PER_MODEL, 0);
        if (workers == 0) {
            workers = Runtime.getRuntime().availableProcessors();
        }
        return workers;
    }

    /**
     * Returns the model server home directory.
     *
     * @return the model server home directory
     */
    public static String getModelServerHome() {
        String home = System.getenv("MODEL_SERVER_HOME");
        if (home == null) {
            home = System.getProperty("MODEL_SERVER_HOME");
            if (home == null) {
                home = getCanonicalPath(".");
                return home;
            }
        }

        Path dir = Paths.get(home);
        if (!Files.isDirectory(dir)) {
            throw new IllegalArgumentException("Model server home not exist: " + home);
        }
        home = getCanonicalPath(dir);
        return home;
    }

    /**
     * Returns the model store location.
     *
     * @return the model store location
     */
    public Path getModelStore() {
        return getPathProperty(MODEL_STORE);
    }

    /**
     * Returns the allowed model url pattern regex.
     *
     * @return the allowed model url pattern regex
     */
    public String getModelUrlPattern() {
        return prop.getProperty(MODEL_URL_PATTERN);
    }

    /**
     * Returns the model urls that to be loaded at startup.
     *
     * @return the model urls that to be loaded at startup
     */
    public String getLoadModels() {
        return prop.getProperty(LOAD_MODELS);
    }

    /**
     * Returns the CORS allowed origin setting.
     *
     * @return the CORS allowed origin setting
     */
    public String getCorsAllowedOrigin() {
        return prop.getProperty(CORS_ALLOWED_ORIGIN);
    }

    /**
     * Returns the CORS allowed method setting.
     *
     * @return the CORS allowed method setting
     */
    public String getCorsAllowedMethods() {
        return prop.getProperty(CORS_ALLOWED_METHODS);
    }

    /**
     * Returns the CORS allowed headers setting.
     *
     * @return the CORS allowed headers setting
     */
    public String getCorsAllowedHeaders() {
        return prop.getProperty(CORS_ALLOWED_HEADERS);
    }

    /**
     * return the folder where the model search for plugins.
     *
     * @return the configured plugin folder or the default folder.
     */
    public Path getPluginFolder() {
        return getPathProperty(PLUGIN_FOLDER, "plugins");
    }

    /**
     * Returns a {@code SSLContext} instance.
     *
     * @return a {@code SSLContext} instance
     * @throws IOException if failed to read certificate file
     * @throws GeneralSecurityException if failed to initialize {@code SSLContext}
     */
    public SslContext getSslContext() throws IOException, GeneralSecurityException {
        List<String> supportedCiphers =
                Arrays.asList(
                        "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA",
                        "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256");

        PrivateKey privateKey;
        X509Certificate[] chain;
        Path keyStoreFile = getPathProperty(KEYSTORE);
        Path privateKeyFile = getPathProperty(PRIVATE_KEY_FILE);
        Path certificateFile = getPathProperty(CERTIFICATE_FILE);
        if (keyStoreFile != null) {
            char[] keystorePass = getProperty(KEYSTORE_PASS, "changeit").toCharArray();
            String keystoreType = getProperty(KEYSTORE_TYPE, "PKCS12");
            KeyStore keyStore = KeyStore.getInstance(keystoreType);
            try (InputStream is = Files.newInputStream(keyStoreFile)) {
                keyStore.load(is, keystorePass);
            }

            Enumeration<String> en = keyStore.aliases();
            String keyAlias = null;
            while (en.hasMoreElements()) {
                String alias = en.nextElement();
                if (keyStore.isKeyEntry(alias)) {
                    keyAlias = alias;
                    break;
                }
            }

            if (keyAlias == null) {
                throw new KeyException("No key entry found in keystore.");
            }

            privateKey = (PrivateKey) keyStore.getKey(keyAlias, keystorePass);

            Certificate[] certs = keyStore.getCertificateChain(keyAlias);
            chain = new X509Certificate[certs.length];
            for (int i = 0; i < certs.length; ++i) {
                chain[i] = (X509Certificate) certs[i];
            }
        } else if (privateKeyFile != null && certificateFile != null) {
            privateKey = loadPrivateKey(privateKeyFile);
            chain = loadCertificateChain(certificateFile);
        } else {
            SelfSignedCertificate ssc = new SelfSignedCertificate();
            privateKey = ssc.key();
            chain = new X509Certificate[] {ssc.cert()};
        }

        return SslContextBuilder.forServer(privateKey, chain)
                .protocols("TLSv1.2")
                .ciphers(supportedCiphers)
                .build();
    }

    /**
     * Returns the value with the specified key in this configuration.
     *
     * @param key the key
     * @param def a default value
     * @return the value with the specified key in this configuration
     */
    public String getProperty(String key, String def) {
        return prop.getProperty(key, def);
    }

    /**
     * Prints out this configuration.
     *
     * @return a string representation of this configuration
     */
    public String dumpConfigurations() {
        Runtime runtime = Runtime.getRuntime();
        return "\nModel server home: "
                + getModelServerHome()
                + "\nCurrent directory: "
                + getCanonicalPath(".")
                + "\nTemp directory: "
                + System.getProperty("java.io.tmpdir")
                + "\nNumber of CPUs: "
                + runtime.availableProcessors()
                + "\nMax heap size: "
                + (runtime.maxMemory() / 1024 / 1024)
                + "\nConfig file: "
                + prop.getProperty("configFile", "N/A")
                + "\nInference address: "
                + getConnector(Connector.ConnectorType.INFERENCE)
                + "\nManagement address: "
                + getConnector(Connector.ConnectorType.MANAGEMENT)
                + "\nModel Store: "
                + (getModelStore() == null ? "N/A" : getModelStore())
                + "\nInitial Models: "
                + (getLoadModels() == null ? "N/A" : getLoadModels())
                + "\nNetty threads: "
                + getNettyThreads()
                + "\nDefault workers per model: "
                + getDefaultWorkers()
                + "\nMaximum Request Size: "
                + prop.getProperty(MAX_REQUEST_SIZE, "6553500");
    }

    /**
     * Returns if use netty native IO.
     *
     * @return {@code true} if use netty native IO
     */
    public boolean useNativeIo() {
        return Boolean.parseBoolean(prop.getProperty(USE_NATIVE_IO, "true"));
    }

    /**
     * Returns the native IO ratio.
     *
     * @return the native IO ratio
     */
    public int getIoRatio() {
        return getIntProperty(IO_RATIO, 50);
    }

    /**
     * Returns the maximum allowed request size in bytes.
     *
     * @return the maximum allowed request size in bytes
     */
    public int getMaxRequestSize() {
        return getIntProperty(MAX_REQUEST_SIZE, 6553500);
    }

    private int getIntProperty(String key, int def) {
        String value = prop.getProperty(key);
        if (value == null) {
            return def;
        }
        return Integer.parseInt(value);
    }

    private Path getPathProperty(String key) {
        return getPathProperty(key, null);
    }

    private Path getPathProperty(String key, String defaultValue) {
        String property = prop.getProperty(key, defaultValue);
        if (property == null) {
            return null;
        }
        Path path = Paths.get(property);
        if (!path.isAbsolute()) {
            path = Paths.get(getModelServerHome()).resolve(path);
        }
        return path;
    }

    private static String getCanonicalPath(Path file) {
        try {
            return file.toRealPath().toString();
        } catch (IOException e) {
            return file.toAbsolutePath().toString();
        }
    }

    private static String getCanonicalPath(String path) {
        if (path == null) {
            return null;
        }
        return getCanonicalPath(Paths.get(path));
    }

    private PrivateKey loadPrivateKey(Path keyFile) throws IOException, GeneralSecurityException {
        KeyFactory keyFactory = KeyFactory.getInstance("RSA");
        try (InputStream is = Files.newInputStream(keyFile)) {
            String content = Utils.toString(is);
            content = content.replaceAll("-----(BEGIN|END)( RSA)? PRIVATE KEY-----\\s*", "");
            byte[] buf = Base64.getMimeDecoder().decode(content);
            try {
                PKCS8EncodedKeySpec privKeySpec = new PKCS8EncodedKeySpec(buf);
                return keyFactory.generatePrivate(privKeySpec);
            } catch (InvalidKeySpecException e) {
                // old private key is OpenSSL format private key
                buf = OpenSslKey.convertPrivateKey(buf);
                PKCS8EncodedKeySpec privKeySpec = new PKCS8EncodedKeySpec(buf);
                return keyFactory.generatePrivate(privKeySpec);
            }
        }
    }

    private X509Certificate[] loadCertificateChain(Path keyFile)
            throws IOException, GeneralSecurityException {
        CertificateFactory cf = CertificateFactory.getInstance("X.509");
        try (InputStream is = Files.newInputStream(keyFile)) {
            Collection<? extends Certificate> certs = cf.generateCertificates(is);
            int i = 0;
            X509Certificate[] chain = new X509Certificate[certs.size()];
            for (Certificate cert : certs) {
                chain[i++] = (X509Certificate) cert;
            }
            return chain;
        }
    }
}
