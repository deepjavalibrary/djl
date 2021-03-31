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
import java.beans.BeanInfo;
import java.beans.IntrospectionException;
import java.beans.Introspector;
import java.beans.PropertyDescriptor;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
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
public class FolderScanPluginManager implements PluginManager {
    private static final Logger logger = LoggerFactory.getLogger(FolderScanPluginManager.class);

    private ConfigManager configManager;

    private Map<Class<?>, Set<Plugin<?>>> pluginsRegistry;

    /**
     * constructing a PluginManager.
     *
     * @param configManager a instance of the configManager to lookup configuration like
     *     plugin-folder.
     */
    public FolderScanPluginManager(ConfigManager configManager) {
        this.configManager = configManager;
        this.pluginsRegistry = new HashMap<>();
    }

    /**
     * loads all plugins from the plugin folder and register them.
     *
     * @throws IOException when error during IO operation occurs.
     */
    @SuppressWarnings("rawtypes")
    public void loadPlugins() throws IOException {
        logger.info("scanning for plugins...");
        URL[] pluginUrls = listPluginJars().toArray(new URL[] {});

        URLClassLoader ucl = new URLClassLoader(pluginUrls);

        int pluginsFound = 0;

        ServiceLoader<RequestHandler> sl = ServiceLoader.load(RequestHandler.class, ucl);

        Iterator<RequestHandler> apit = sl.iterator();
        while (apit.hasNext()) {
            pluginsFound++;
            RequestHandler<?> service = apit.next();
            logger.info("load plugin: {}", service.getClass().getSimpleName());
            Plugin<RequestHandler<?>> plugin = new Plugin<>(service);
            // TODO add a plugin Lifecycle "INITIALIZING", "ACTIVE", "SHUTTING DOWN" , so a plug-in
            // can be dependent on another plugin.
            if (initializePlugin(plugin)) {
                pluginsRegistry
                        .computeIfAbsent(RequestHandler.class, k -> new HashSet<Plugin<?>>())
                        .add(plugin);
            }
        }
        logger.info("{} plug-ins found.", pluginsFound);
    }

    /**
     * initialize a plugin by calling known setters to inject managers and other dependant plugins
     * into the plugins
     *
     * <p>initializePlugin.
     *
     * @param plugin the plugin to get initialized
     * @return true if plugin could get initialized successfully false otherwise
     */
    private boolean initializePlugin(Plugin<?> plugin) {
        Object component = plugin.getComponent();
        try {
            BeanInfo beanInfo = Introspector.getBeanInfo(component.getClass());
            for (PropertyDescriptor property : beanInfo.getPropertyDescriptors()) {
                // TODO introduce kind of ServiceRegistry and inject all known Managers and others
                // plug-ins
                if ("pluginManager".equals(property.getName())) {
                    Method method = property.getWriteMethod();
                    if (method != null) {
                        method.invoke(component, this);
                    } else {
                        logger.warn(
                                "no accessible setter for pluginManager found in plugin {}. skipping injecting",
                                plugin.getName());
                    }
                }
            }
        } catch (IntrospectionException
                | IllegalAccessException
                | IllegalArgumentException
                | InvocationTargetException e) {
            logger.error(
                    "plugin {} could not get loaded. Initialization failed", plugin.getName(), e);
            return false;
        }
        return true;
    }

    /**
     * returns a set of plug-in components implementing the specific service interface.
     *
     * <p>only active plug-ins are returned which are fully initialised at this point.
     *
     * <p>{@code Set<RequestHandler>
     * allActiveRequestHandler=findImplementations(RequestHandler.class)}
     *
     * @param <T> generic type of service interface
     * @param pluginInterface the specific service interface
     * @return a set of all plugin components implementing this service interface
     */
    @Override
    public <T> Set<T> findImplementations(Class<T> pluginInterface) {
        return Collections.unmodifiableSet(
                pluginsRegistry
                        .getOrDefault(pluginInterface, new HashSet<Plugin<?>>())
                        .stream()
                        .map(Plugin::getComponent)
                        .map(pluginInterface::cast)
                        .collect(Collectors.toSet()));
    }

    @SuppressWarnings("unchecked")
    private Set<URL> listPluginJars() throws IOException {
        Path pluginsFolder = configManager.getPluginFolder();
        if (!(Files.exists(pluginsFolder) && Files.isDirectory(pluginsFolder))) {
            logger.warn("scanning in plug-in folder :{}....folder does not exists", pluginsFolder);
            return (Set<URL>) Collections.EMPTY_SET;
        }
        logger.debug("scanning in plug-in folder :{}", pluginsFolder);

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

    // TODO maybe extract this to a public class in serving-api, so we can have functions like
    // "listPlugin" which return Plugin objects
    class Plugin<T> {
        private T component;
        private LocalDateTime loadtime;

        public Plugin(T component) {
            this.component = component;
            this.loadtime = LocalDateTime.now();
        }

        /**
         * gets the value of component.
         *
         * @return the component value.
         */
        public T getComponent() {
            return component;
        }

        /**
         * gets the value of loadtime.
         *
         * @return the loadtime value.
         */
        public LocalDateTime getLoadtime() {
            return loadtime;
        }

        /**
         * return the name of the plug-in.
         *
         * @return name of the plug-in.
         */
        public String getName() {
            return component.getClass().getSimpleName();
        }
    }
}
