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
package ai.djl.repository;

/**
 * A {@code License} is a container to save the license information.
 *
 * @see Repository
 */
public class License {
    private transient String id;

    private String name;
    private String url;

    /**
     * The default Apache License.
     *
     * @return Apache license
     */
    public static License apache() {
        License license = new License();
        license.setName("The Apache License, Version 2.0");
        license.setUrl("https://www.apache.org/licenses/LICENSE-2.0");
        license.setId("apache");
        return license;
    }

    /**
     * Returns the name of the license.
     *
     * @return the name of the license
     */
    public String getName() {
        return name;
    }

    /**
     * Sets the name of the license.
     *
     * @param name the name of the license
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * Returns the url of the license.
     *
     * @return the url of the license
     */
    public String getUrl() {
        return url;
    }

    /**
     * Sets the url of the license.
     *
     * @param url the url of the license;
     */
    public void setUrl(String url) {
        this.url = url;
    }

    /**
     * Sets the identifier of the license.
     *
     * @param id the identifier of the license.
     */
    public void setId(String id) {
        this.id = id;
    }

    /**
     * Returns the identifier of the license.
     *
     * @return the identifier of the license
     */
    public String getId() {
        return id;
    }
}
