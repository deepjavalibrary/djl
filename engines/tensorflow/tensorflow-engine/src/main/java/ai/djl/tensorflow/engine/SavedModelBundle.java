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

import org.tensorflow.internal.c_api.TF_Graph;
import org.tensorflow.internal.c_api.TF_Operation;
import org.tensorflow.internal.c_api.TF_Session;
import org.tensorflow.internal.c_api.global.tensorflow;
import org.tensorflow.proto.framework.CollectionDef;
import org.tensorflow.proto.framework.MetaGraphDef;
import org.tensorflow.proto.framework.SignatureDef;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/** The wrapper class for native resources required for SavedModelBundle. */
public class SavedModelBundle implements AutoCloseable {

    private static final String INIT_OP_SIGNATURE_KEY = "__saved_model_init_op";
    private static final String MAIN_OP_COLLECTION_KEY = "saved_model_main_op";
    private static final String LEGACY_INIT_OP_COLLECTION_KEY = "legacy_init_op";
    private static final String TABLE_INITIALIZERS_COLLECTION_KEY = "table_initializer";

    private TF_Graph graphHandle;
    private TF_Session sessionHandle;
    private MetaGraphDef metaGraphDef;
    private TF_Operation[] targetOpHandles;
    private AtomicBoolean closed;

    public SavedModelBundle(
            TF_Graph graphHandle, TF_Session sessionHandle, MetaGraphDef metaGraphDef) {
        this.graphHandle = graphHandle;
        this.sessionHandle = sessionHandle;
        this.metaGraphDef = metaGraphDef;
        closed = new AtomicBoolean(false);

        Map<String, SignatureDef> functions = new ConcurrentHashMap<>();
        metaGraphDef
                .getSignatureDefMap()
                .forEach(
                        (signatureName, signatureDef) -> {
                            if (!functions.containsKey(signatureName)) {
                                functions.put(signatureName, signatureDef);
                            }
                        });

        List<TF_Operation> initOps = new ArrayList<>();
        TF_Operation initOp = findInitOp(functions, metaGraphDef.getCollectionDefMap());
        if (initOp != null) {
            initOps.add(initOp);
        }

        if (metaGraphDef.containsCollectionDef(TABLE_INITIALIZERS_COLLECTION_KEY)) {
            metaGraphDef
                    .getCollectionDefMap()
                    .get(TABLE_INITIALIZERS_COLLECTION_KEY)
                    .getNodeList()
                    .getValueList()
                    .forEach(
                            node -> {
                                initOps.add(tensorflow.TF_GraphOperationByName(graphHandle, node));
                            });
        }
        targetOpHandles = initOps.toArray(new TF_Operation[0]);
    }

    private TF_Operation findInitOp(
            Map<String, SignatureDef> signatures, Map<String, CollectionDef> collections) {
        SignatureDef initSig = signatures.get(INIT_OP_SIGNATURE_KEY);
        if (initSig != null) {
            String opName = initSig.getOutputsMap().get(INIT_OP_SIGNATURE_KEY).getName();
            return tensorflow.TF_GraphOperationByName(graphHandle, opName);
        }

        CollectionDef initCollection;
        if (collections.containsKey(MAIN_OP_COLLECTION_KEY)) {
            initCollection = collections.get(MAIN_OP_COLLECTION_KEY);
        } else {
            initCollection = collections.get(LEGACY_INIT_OP_COLLECTION_KEY);
        }

        if (initCollection != null) {
            CollectionDef.NodeList nodes = initCollection.getNodeList();
            if (nodes.getValueCount() != 1) {
                throw new IllegalArgumentException("Expected exactly one main op in saved model.");
            }
            String opName = nodes.getValue(0);
            return tensorflow.TF_GraphOperationByName(graphHandle, opName);
        }
        return null;
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

    TF_Operation[] getTargetOpHandles() {
        return targetOpHandles;
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
