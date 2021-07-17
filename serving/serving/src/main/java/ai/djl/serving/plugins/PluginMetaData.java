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
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Represents a loaded Plug-in.
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
    }

    private String name;
    private URL url;
    private List<String> componentNames;
    private List<String> dependencies;
    private Lifecycle state;
    private String error;

    /**
     * Constructs a plug-in meta-info.
     *
     * @param name of the plug-in
     * @param url where this plug-in is loaded from
     * @param componentNames of all exported components of the plug-in
     * @param dependencies require this plug-ins to run
     */
    public PluginMetaData(
            String name, URL url, List<String> componentNames, List<String> dependencies) {
        this.name = name;
        this.url = url;
        this.componentNames = componentNames;
        this.state = Lifecycle.INITIALIZING;
        this.dependencies = dependencies;
    }

    /**
     * Returns the name of the plug-in.
     *
     * @return name of the plug-in
     */
    public String getName() {
        return name;
    }

    /**
     * Returns the class names of the registered-components.
     *
     * @return the class names of the registered-components
     */
    public List<String> getComponentNames() {
        return Collections.unmodifiableList(componentNames);
    }

    /**
     * Returns the state-property.
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
        logger.debug("plugin {} changed state to {}", name, state);
    }

    /**
     * Sets the property state of the object and log message.
     *
     * @param state the state to set
     * @param logMessage why this status is set
     */
    public void changeState(Lifecycle state, String logMessage) {
        this.state = state;

        if (state == Lifecycle.FAILED) {
            error = logMessage;
            logger.warn("plugin {} changed state to {} reason: {}", name, state, logMessage);
        } else {
            logger.debug("plugin {} changed state to {} reason: {}", name, state, logMessage);
        }
    }

    /**
     * Returns the url-property.
     *
     * @return the url of this class
     */
    public URL getUrl() {
        return url;
    }

    /**
     * Returns a list of required plug-in dependencies.
     *
     * @return the depend plug-in names require by this class
     */
    public List<String> getDependencies() {
        return dependencies;
    }

    /**
     * Returns the error-property.
     *
     * @return the error of this class
     */
    public String getError() {
        return error;
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof PluginMetaData)) {
            return false;
        }
        PluginMetaData that = (PluginMetaData) o;
        return name.equals(that.name);
    }

    /** {@inheritDoc} */
    @Override
    public int hashCode() {
        return Objects.hash(name);
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return '{' + name + '/' + url + '}';
    }
}
