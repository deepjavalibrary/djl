package ai.djl.tensorflow.engine;

import static org.tensorflow.internal.c_api.global.tensorflow.*;

import java.util.concurrent.atomic.AtomicBoolean;
import org.bytedeco.javacpp.PointerScope;
import org.tensorflow.internal.c_api.TF_Graph;
import org.tensorflow.internal.c_api.TF_Session;
import org.tensorflow.internal.c_api.TF_Status;
import org.tensorflow.proto.framework.MetaGraphDef;

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

    public TF_Graph getGraph() {
        return graphHandle;
    }

    public TF_Session getSession() {
        return sessionHandle;
    }

    public MetaGraphDef getMetaGraphDef() {
        return metaGraphDef;
    }

    @SuppressWarnings({"unchecked", "try"})
    @Override
    public void close() {
        // to prevent double free
        if (closed.getAndSet(true)) {
            return;
        }
        if (graphHandle != null && !graphHandle.isNull()) {
            graphHandle.close();
        }
        // manually use C API to release session native resource
        // as the default session's deallocator in org/tensorflow/c_api/internal/TF_Session.java
        // have not released status and called throwExceptionIfNotOK
        // usually we should prefer to use default deallocator as much as possible
        try (PointerScope scope = new PointerScope()) {
            TF_Status status = TF_Status.newStatus();
            TF_CloseSession(sessionHandle, status);
            // Result of close is ignored, delete anyway.
            TF_DeleteSession(sessionHandle, status);
            status.throwExceptionIfNotOK();
        }
        metaGraphDef = null;
    }
}
