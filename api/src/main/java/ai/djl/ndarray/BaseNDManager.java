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
package ai.djl.ndarray;

import ai.djl.Device;
import java.lang.ref.Reference;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** {@code BaseNDManager} is the default implementation of {@link NDManager}. */
public abstract class BaseNDManager implements NDManager {

    private static final Logger logger = LoggerFactory.getLogger(BaseNDManager.class);

    protected NDManager parent;
    protected String uid;
    protected Device device;
    protected Map<String, Reference<AutoCloseable>> resources;
    protected AtomicBoolean closed = new AtomicBoolean(false);

    protected BaseNDManager(NDManager parent, Device device) {
        this.parent = parent;
        this.device = Device.defaultIfNull(device);
        resources = new ConcurrentHashMap<>();
        uid = UUID.randomUUID().toString();
    }

    /** {@inheritDoc} */
    @Override
    public boolean isOpen() {
        return !closed.get();
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getParentManager() {
        return parent;
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        return device;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        String parentUID = parent == null ? "No Parent" : ((BaseNDManager) parent).uid;
        return "UID: "
                + uid
                + " Parent UID: "
                + parentUID
                + " isOpen: "
                + isOpen()
                + " Resource size: "
                + resources.size();
    }

    /** {@inheritDoc} */
    @Override
    public synchronized void close() {
        if (!closed.getAndSet(true)) {
            for (Reference<AutoCloseable> resource : resources.values()) {
                AutoCloseable closeable = resource.get();
                if (closeable != null) {
                    try {
                        closeable.close();
                    } catch (Exception e) {
                        logger.error("Resource close failed.", e);
                    }
                }
            }
            parent.detach(uid);
            resources.clear();
        }
    }
}
