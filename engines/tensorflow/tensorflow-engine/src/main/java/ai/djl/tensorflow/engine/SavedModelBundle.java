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

package ai.djl.tensorflow.engine;

import java.util.concurrent.atomic.AtomicBoolean;
import org.tensorflow.internal.c_api.TF_Graph;
import org.tensorflow.internal.c_api.TF_Session;
import org.tensorflow.proto.framework.MetaGraphDef;

/** The wrapper class for native resources required for SavedModelBundle. */
public class SavedModelBundle implements AutoCloseable {

    private TF_Graph graphHandle;
    private TF_Session sessionHandle;
    private MetaGraphDef metaGraphDef;
    private AtomicBoolean closed;

    public SavedModelBundle(
            TF_Graph graphHandle, TF_Session sessionHandle, MetaGraphDef metaGraphDef) {
        this.graphHandle = graphHandle;
        this.sessionHandle = sessionHandle;
        this.metaGraphDef = metaGraphDef;
        closed = new AtomicBoolean(false);
    }

    /**
     * Returns the graph handle.
     *
     * @return the graph handle
     */
    public TF_Graph getGraph() {
        return graphHandle;
    }

    /**
     * Returns the session handle.
     *
     * @return the session handle
     */
    public TF_Session getSession() {
        return sessionHandle;
    }

    /**
     * Returns the MetaGraphDef protol buf.
     *
     * @return the MetaGraphDef protol buf
     */
    public MetaGraphDef getMetaGraphDef() {
        return metaGraphDef;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        // to prevent double free
        if (closed.getAndSet(true)) {
            return;
        }
        if (graphHandle != null && !graphHandle.isNull()) {
            graphHandle.close();
        }
        if (sessionHandle != null && !sessionHandle.isNull()) {
            sessionHandle.close();
        }
        metaGraphDef = null;
    }
}
