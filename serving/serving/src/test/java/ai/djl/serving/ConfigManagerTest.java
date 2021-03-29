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
package ai.djl.serving;

import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.Connector;
import io.netty.handler.ssl.util.SelfSignedCertificate;
import java.io.IOException;
import java.io.OutputStream;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.security.GeneralSecurityException;
import java.security.KeyStore;
import java.security.PrivateKey;
import java.security.cert.Certificate;
import java.security.cert.X509Certificate;
import java.util.Properties;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.testng.Assert;

public final class ConfigManagerTest {

    private ConfigManagerTest() {}

    public static void testSsl()
            throws IOException, GeneralSecurityException, ParseException,
                    ReflectiveOperationException {
        ConfigManager.init(parseArguments(new String[0]));
        ConfigManager config = ConfigManager.getInstance();
        Assert.assertNotNull(config.getSslContext());
        Assert.assertNotNull(ConfigManager.getModelServerHome());
        Assert.assertEquals(config.getIoRatio(), 50);

        setConfiguration(config, "keystore", "build/tmp/keystore.jks");
        setConfiguration(config, "keystore_pass", "changeit");
        setConfiguration(config, "keystore_type", "JKS");

        Path dir = Paths.get("build/tmp");
        Files.createDirectories(dir);
        Path ksFile = Paths.get("build/tmp/keystore.jks");
        if (Files.notExists(ksFile)) {
            SelfSignedCertificate selfSign = new SelfSignedCertificate();
            X509Certificate cert = selfSign.cert();
            PrivateKey key = selfSign.key();
            Certificate[] chain = {cert};

            KeyStore ks = KeyStore.getInstance("JKS");
            ks.load(null);
            ks.setKeyEntry("djl", key, "changeit".toCharArray(), chain);
            try (OutputStream os = Files.newOutputStream(ksFile, StandardOpenOption.CREATE)) {
                ks.store(os, "changeit".toCharArray());
            }
            selfSign.delete();
        }

        config = ConfigManager.getInstance();
        config.getSslContext();

        Assert.assertEquals(
                ConfigManager.getModelServerHome(), Paths.get(".").toRealPath().toString());

        config.getConnector(Connector.ConnectorType.INFERENCE);
    }

    public static Arguments parseArguments(String[] args) throws ParseException {
        Options options = Arguments.getOptions();
        DefaultParser parser = new DefaultParser();
        CommandLine cmd = parser.parse(options, args, null, false);
        return new Arguments(cmd);
    }

    public static void setConfiguration(ConfigManager configManager, String key, String val)
            throws NoSuchFieldException, IllegalAccessException {
        Field f = configManager.getClass().getDeclaredField("prop");
        f.setAccessible(true);
        Properties p = (Properties) f.get(configManager);
        p.setProperty(key, val);
    }
}
