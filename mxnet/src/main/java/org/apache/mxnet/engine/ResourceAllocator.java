/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package org.apache.mxnet.engine;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class ResourceAllocator implements java.io.Closeable {

    private Map<NativeResource, NativeResource> resources;

    public ResourceAllocator() {
        resources = new ConcurrentHashMap<>();
    }

    public synchronized void attach(NativeResource resource) {
        resources.put(resource, resource);
    }

    public synchronized void detach(NativeResource resource) {
        resources.remove(resource);
    }

    public synchronized void detach(NativeResource resource, boolean close) {
        resources.remove(resource);
    }

    @Override
    public synchronized void close() {
        for (NativeResource resource : resources.keySet()) {
            resource.close();
        }
        resources = null;
    }
}
