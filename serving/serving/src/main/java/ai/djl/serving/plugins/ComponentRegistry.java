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
import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * A registry for plugin components.
 *
 * @author erik.bamberg@web.de
 */
public class ComponentRegistry {

    private static Class<?>[] pluginInterfaces = {RequestHandler.class};

    private Map<Class<?>, Set<ComponentEntry>> componentRegistry;

    /** construct a registry. */
    public ComponentRegistry() {
        componentRegistry = new ConcurrentHashMap<>();
    }

    /**
     * Registers a new component and assign the plug-in as source of this component.
     *
     * @param plugin which this component is linked to
     * @param component the component
     */
    public void register(PluginMetaData plugin, RequestHandler<?> component) {
        for (Class<?> interfaceClass : pluginInterfaces) {
            if (interfaceClass.isAssignableFrom(component.getClass())) {
                componentRegistry
                        .computeIfAbsent(interfaceClass, k -> new HashSet<>())
                        .add(new ComponentEntry(plugin, component));
            }
        }
    }

    /**
     * Returns a set of plug-in components implementing the specific service interface.
     *
     * <p>Only active plug-ins are returned which are fully initialised at this point.
     *
     * <p>{@code Set<RequestHandler>
     * allActiveRequestHandler=findImplementations(RequestHandler.class)}
     *
     * @param <T> generic type of service interface
     * @param pluginInterface the specific service interface
     * @return a set of all plugin components implementing this service interface
     */
    @SuppressWarnings("unchecked")
    public <T> Set<T> findImplementations(Class<T> pluginInterface) {
        return (Set<T>)
                Collections.unmodifiableSet(
                        componentRegistry
                                .getOrDefault(pluginInterface, new HashSet<>())
                                .stream()
                                .filter(ComponentEntry::isPluginActive)
                                .map(ComponentEntry::getComponent)
                                .collect(Collectors.toSet()));
    }

    private static class ComponentEntry {
        private PluginMetaData plugin;
        private Object component;

        public ComponentEntry(PluginMetaData plugin, Object component) {
            super();
            this.plugin = plugin;
            this.component = component;
        }

        public boolean isPluginActive() {
            return plugin.getState() == Lifecycle.ACTIVE;
        }

        public Object getComponent() {
            return component;
        }
    }
}
