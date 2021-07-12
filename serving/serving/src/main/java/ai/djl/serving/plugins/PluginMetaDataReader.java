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

/**
 * Reads plugin-metadata from an url and parse the content.
 *
 * <p>Implementations typically reads a definition file like {@code plugin.definion} from the plugin
 * jar file.
 *
 * @author erik.bamberg@web.de
 */
public interface PluginMetaDataReader {

    /**
     * Reads plugin-metadata from on url.
     *
     * @return a parsed plugin metadata object or null if not metadata can be found
     */
    PluginMetaData read();
}
