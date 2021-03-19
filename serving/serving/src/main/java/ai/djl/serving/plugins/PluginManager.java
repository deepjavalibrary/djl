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

import ai.djl.serving.util.ConfigManager;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The Plugin Manager is responsible to load and manage plugins from the filesystem.
 *
 * <p>The Plugin Folder configuration is received from the ConfigManager and usually defaults to
 * {workpath}/plugins. The plugins uses Java's SPI and have to implement interfaces from
 * serving-api.
 *
 * @author erik.bamberg@web.de
 */
public class PluginManager {
    private static final Logger logger = LoggerFactory.getLogger(PluginManager.class);

    private ConfigManager configManager;
    
    /** constructing a PluginManager. */
    public PluginManager(ConfigManager configManager) {
	this.configManager=configManager;
    }

    /**
     * loads all plugins from the plugin folder and register them.
     *
     * @throws IOException when error during IO operation occurs.
     */
    public void loadPlugins() throws IOException {
	logger.info("scanning for plugins...");
        URL[] pluginUrls = listPluginJars().toArray(new URL[] {});
        
        URLClassLoader ucl = new URLClassLoader(pluginUrls);

        int pluginsFound=0;
        ServiceLoader<RequestHandler> sl = ServiceLoader.load(RequestHandler.class, ucl);
        Iterator<RequestHandler> apit = sl.iterator();
        while (apit.hasNext()) {
            pluginsFound++;
            logger.debug("load plugin: {}",apit.next().getClass().getSimpleName());
        }
        logger.info("{} plug-ins found.",pluginsFound);
    }

    private Set<URL> listPluginJars() throws IOException {
        Path pluginsFolder = configManager.getPluginFolder();
        logger.debug("scanning in plug-in folder :{}",pluginsFolder);
        
        try (Stream<Path> stream = Files.walk(pluginsFolder, Integer.MAX_VALUE)) {
            return stream.filter(file -> !Files.isDirectory(file))
                    .filter(file -> file.getFileName() != null)
                    .filter(file -> file.getFileName().toString().toLowerCase().endsWith(".jar"))
                    .map(Path::toUri)
                    .map(
                            t -> {
                                try {
                                    return t.toURL();
                                } catch (MalformedURLException e) {
                                    logger.error(e.getMessage(), e);
                                }
                                return null;
                            })
                    .collect(Collectors.toSet());
        }
    }
}
