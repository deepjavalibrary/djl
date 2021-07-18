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

import ai.djl.serving.plugins.PluginMetaData.Lifecycle;
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
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The {@link PluginManager} is responsible to load and manage plugins from the file system.
 *
 * <p>The Plugin Folder configuration is received from the {@link ConfigManager} and usually
 * defaults to {workpath}/plugins. The plugins uses Java's SPI and have to implement interfaces from
 * serving-api.
 *
 * @author erik.bamberg@web.de
 */
public class FolderScanPluginManager implements PluginManager {

    private static final Logger logger = LoggerFactory.getLogger(FolderScanPluginManager.class);

    private ConfigManager configManager;

    private Map<String, PluginMetaData> pluginRegistry;

    private ComponentRegistry componentRegistry;

    /**
     * Constructs a {@code PluginManager} instance.
     *
     * @param configManager a instance of the configManager to lookup configuration like
     *     plugin-folder.
     */
    public FolderScanPluginManager(ConfigManager configManager) {
        this.configManager = configManager;
        this.componentRegistry = new ComponentRegistry();
    }

    /**
     * Loads all plugins from the plugin folder and register them.
     *
     * @throws IOException when error during IO operation occurs.
     */
    public void loadPlugins() throws IOException {
        logger.info("scanning for plugins...");
        URL[] pluginUrls = listPluginJars();

        ClassLoader ucl =
                AccessController.doPrivileged(
                        (PrivilegedAction<ClassLoader>) () -> new URLClassLoader(pluginUrls));

        // phase 1: collect plugin information
        pluginRegistry =
                Collections.list(ucl.getResources("META-INF/plugin.definition"))
                        .parallelStream()
                        .map(PropertyFilePluginMetaDataReader::new)
                        .map(PropertyFilePluginMetaDataReader::read)
                        .distinct()
                        .collect(Collectors.toMap(PluginMetaData::getName, i -> i));

        // phase 2: initialize components
        for (PluginMetaData plugin : pluginRegistry.values()) {
            logger.info("Loading plugin: {}", plugin);
            if (pluginRegistry.keySet().containsAll(plugin.getDependencies())) {
                try {
                    for (String handlerClassName : plugin.getComponentNames()) {
                        initializeComponent(ucl, plugin, handlerClassName);
                    }
                    plugin.changeState(Lifecycle.INITIALIZED);
                } catch (Throwable t) {
                    plugin.changeState(
                            Lifecycle.FAILED,
                            "failed to initialize plugin; caused by " + t.getMessage());
                    logger.error("failed to initialize plugin {}", plugin.getName(), t);
                }
            } else {
                plugin.changeState(Lifecycle.FAILED, "required dependencies not found");
            }
        }

        // phase 3: set active
        pluginRegistry
                .values()
                .stream()
                .filter(plugin -> plugin.getState() == Lifecycle.INITIALIZED)
                .filter(this::checkAllRequiredPluginsInitialized)
                .forEach(plugin -> plugin.changeState(Lifecycle.ACTIVE, "plugin ready"));

        logger.info("{} plug-ins found and loaded.", pluginRegistry.size());
    }

    /**
     * Checks if all plug-ins required by this plugin are Initialized.
     *
     * @param plugin to check dependencies for state="Initialized"
     * @return true if all plugins required by this one are in state "Initialized"
     */
    private boolean checkAllRequiredPluginsInitialized(PluginMetaData plugin) {
        for (String required : plugin.getDependencies()) {
            PluginMetaData reqPlugin = pluginRegistry.get(required);
            if (reqPlugin != null && reqPlugin.getState() != Lifecycle.INITIALIZED) {
                return false;
            }
        }
        return true;
    }

    @SuppressWarnings("rawtypes")
    protected void initializeComponent(
            ClassLoader ucl, PluginMetaData plugin, String handlerClassName)
            throws ReflectiveOperationException, IntrospectionException {
        @SuppressWarnings("unchecked")
        Class<? extends RequestHandler> handlerClass =
                (Class<? extends RequestHandler>) ucl.loadClass(handlerClassName);
        RequestHandler<?> handler = handlerClass.getConstructor(new Class<?>[] {}).newInstance();
        injectDependenciesIntoComponent(handler);

        componentRegistry.register(plugin, handler);
    }

    /**
     * Initializes a plugin by calling known setters to inject managers and other dependant plugins
     * into the plugins.
     *
     * @param component the component to get initialized
     * @throws IntrospectionException when initialization fails.
     * @throws InvocationTargetException when initialization fails.
     * @throws ReflectiveOperationException when initialization fails.
     */
    protected void injectDependenciesIntoComponent(Object component)
            throws IntrospectionException, ReflectiveOperationException, InvocationTargetException {
        BeanInfo beanInfo = Introspector.getBeanInfo(component.getClass());
        for (PropertyDescriptor property : beanInfo.getPropertyDescriptors()) {
            // TODO introduce kind of ServiceRegistry and inject all known Managers and
            // others
            // plug-ins
            if ("pluginManager".equals(property.getName())) {
                Method method = property.getWriteMethod();
                if (method != null) {
                    method.invoke(component, this);
                } else {
                    logger.warn(
                            "no accessible setter for pluginManager found in plugin {}. skipping injecting",
                            component.getClass().getName());
                }
            }
        }
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
        return componentRegistry.findImplementations(pluginInterface);
    }

    private URL[] listPluginJars() throws IOException {
        Path pluginsFolder = configManager.getPluginFolder();
        if (pluginsFolder == null || !Files.isDirectory(pluginsFolder)) {
            logger.warn("scanning in plug-in folder :{}....folder does not exists", pluginsFolder);
            return new URL[0];
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
                    .toArray(URL[]::new);
        }
    }

    /**
     * List all plugins.
     *
     * @return list of all plugins.
     */
    @Override
    public Collection<PluginMetaData> listPlugins() {
        return Collections.unmodifiableCollection(pluginRegistry.values());
    }
}
