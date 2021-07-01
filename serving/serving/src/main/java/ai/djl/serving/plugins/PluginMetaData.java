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

import java.net.URL;
import java.time.LocalDateTime;
import java.util.Collections;
import java.util.List;
import java.util.function.BooleanSupplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * represents a loaded Plug-in.
 *
 * <p>A plug-in contains MetaData, handler and resource mapping information and references to the
 * plug-in components
 *
 * @author erik.bamberg@web.de
 */
public class PluginMetaData {
    private static final Logger logger = LoggerFactory.getLogger(PluginMetaData.class);

    enum Lifecycle {
        INITIALIZING,
        INITIALIZED,
        ACTIVE,
        INACTIVE,
        FAILED
    };

    private String name;
    private LocalDateTime loadTime;
    private URL url;
    private List<String> componentNames;
    private List<String> dependencies;
    private Lifecycle state;
    //   private List<Object> componentRegistry;

    /**
     * constructs plug-in meta-info.
     *
     * @param name of the plug-in.
     * @param url where this plug-in is loaded from.
     * @param componentNames of all exported components of the plug-in.
     * @param dependencies require this plug-ins to run.
     */
    public PluginMetaData(
            String name, URL url, List<String> componentNames, List<String> dependencies) {
        this.name = name;
        this.url = url;
        this.componentNames = componentNames;
        this.loadTime = LocalDateTime.now();
        this.state = Lifecycle.INITIALIZING;
        this.dependencies = dependencies;
    }

    /**
     * Returns the value of loadtime.
     *
     * @return the loadtime value.
     */
    public LocalDateTime getLoadTime() {
        return loadTime;
    }

    /**
     * Returns the name of the plug-in.
     *
     * @return name of the plug-in.
     */
    public String getName() {
        return name;
    }

    /**
     * Return the classnames of the registered-components.
     *
     * @return the classnames of the registered-components.
     */
    public List<String> getComponents() {
        return Collections.unmodifiableList(componentNames);
    }

    /**
     * Access the state-property.
     *
     * @return the state of this class.
     */
    public Lifecycle getState() {
        return state;
    }

    /**
     * Sets the property state of the object.
     *
     * @param state the state to set
     */
    public void changeState(Lifecycle state) {
        this.state = state;
    }

    /**
     * Sets the property state of the object and log message.
     *
     * @param state the state to set
     * @param logMessage why this status is set
     */
    public void changeState(Lifecycle state, String logMessage) {
        this.state = state;
        logger.info("plugin {} changed state to {} reason: {}", name, state, logMessage);
    }

    /**
     * Sets the property state of the object and log message when the lambda returns true.
     *
     * <p>returns the result of the BooleanSupplier.
     *
     * @param predicate to test if we should change the state
     * @param state the state to set
     * @param logMessage why this status is set
     * @return result of the predicate
     */
    public boolean changeStateWhen(BooleanSupplier predicate, Lifecycle state, String logMessage) {
        if (predicate.getAsBoolean()) {
            this.state = state;
            logger.info("plugin {} changed state to {} reason: {}", name, state, logMessage);
            return true;
        } else {
            return false;
        }
    }

    /**
     * Access the url-property.
     *
     * @return the url of this class.
     */
    public URL getUrl() {
        return url;
    }

    /**
     * List required plug-in dependencies.
     *
     * @return the depend plug-in names require by this class.
     */
    public List<String> getDependencies() {
        return dependencies;
    }
}
