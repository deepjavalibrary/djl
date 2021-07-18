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
package ai.djl.serving.plugins;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Properties;
import java.util.UUID;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Reads plugin-metadata from plugin.definition file which have to be in property file format.
 *
 * <pre>
 * name=static-file-plugin
 * requestHandler=ai.djl.serving.plugins.staticfile.HttpStaticClasspathResourceHandler
 * </pre>
 *
 * @author erik.bamberg@web.de
 */
public class PropertyFilePluginMetaDataReader implements PluginMetaDataReader {

    private static final String PROPERTY_PLUGIN_NAME = "name";
    private static final String PROPERTY_PLUGIN_EXPORT = "export";
    private static final String PROPERTY_PLUGIN_REQUIRES = "requires";

    private static final Logger logger =
            LoggerFactory.getLogger(PropertyFilePluginMetaDataReader.class);

    private Properties properties;
    private URL url;

    /**
     * Constructs a {@code PropertyFilePluginMetaDataReader} instance to read meta-information from
     * the URL.
     *
     * @param url to read the plug-in meta-data from
     */
    public PropertyFilePluginMetaDataReader(URL url) {
        this.url = url;
        properties = new Properties();

        try (InputStream is = url.openConnection().getInputStream()) {
            properties.load(is);
        } catch (IOException e) {
            logger.error("io error while receiving plugin.definition file", e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public PluginMetaData read() {
        String pluginName =
                properties.getProperty(
                        PROPERTY_PLUGIN_NAME, "plugin_" + UUID.randomUUID().toString());
        List<String> exportedComponents = getPropertyAsStringList(PROPERTY_PLUGIN_EXPORT);
        List<String> requires = getPropertyAsStringList(PROPERTY_PLUGIN_REQUIRES);

        logger.info("Plugin found: {}/{}", pluginName, url);
        return new PluginMetaData(pluginName, url, exportedComponents, requires);
    }

    private List<String> getPropertyAsStringList(String property) {
        String rhNames = properties.getProperty(property, "");
        List<String> exportedComponents;
        if (!rhNames.isEmpty()) {
            exportedComponents = Arrays.asList(rhNames.split("\\s*,\\s*"));
        } else {
            exportedComponents = Collections.emptyList();
        }
        return exportedComponents;
    }
}
