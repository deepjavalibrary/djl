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

import java.util.Collection;
import java.util.Set;

/**
 * The Plugin Manager is responsible to load and manage plugins from the filesystem.
 *
 * <p>The Plugin Folder configuration is received from the ConfigManager and usually defaults to
 * {workpath}/plugins. The plugins uses Java's SPI and have to implement interfaces from
 * serving-api.
 *
 * @author erik.bamberg@web.de
 */
public interface PluginManager {

    /**
     * Returns a set of plug-in components implementing the specific service interface.
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
    <T> Set<T> findImplementations(Class<T> pluginInterface);

    /**
     * Returns a collection of all plugins registered.
     *
     * @return collection of all registered plugins.
     */
    Collection<PluginMetaData> listPlugins();
}
