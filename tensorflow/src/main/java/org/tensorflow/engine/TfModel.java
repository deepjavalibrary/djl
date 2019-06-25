package org.tensorflow.engine;

import com.amazon.ai.Model;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.DataType;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.function.Function;
import org.tensorflow.Graph;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;

public class TfModel implements Model {

    private Path modelDir;
    private SavedModelBundle bundle;

    TfModel(
            Path modelDir,
            SavedModelBundle bundle) {
        this.modelDir = modelDir;
        this.bundle = bundle;
    }

    static TfModel loadModel(String modelDir, String... tags) {
        if (tags == null || tags.length == 0) {
            tags = new String[]{"serve"};
        }
        return new TfModel(
                Paths.get(modelDir),
                SavedModelBundle.load(modelDir, tags)
        );
    }

    static TfModel loadModel(String modelDir, byte[] configProto, byte[] runOptions, String... tags) {
        SavedModelBundle bundle = SavedModelBundle.loader(modelDir)
                .withConfigProto(configProto).withRunOptions(runOptions).withTags(tags).load();
        return new TfModel(
                Paths.get(modelDir),
                bundle
        );
    }

    public Graph getGraph() {
        return bundle.graph();
    }

    public Session getSession() {
        return bundle.session();
    }

    public byte[] getMetaGraphDef() {
        return bundle.metaGraphDef();
    }

    /** {@inheritDoc} */
    @Override
    public DataDesc[] describeInput() {
        return new DataDesc[0];
    }

    /** {@inheritDoc} */
    @Override
    public DataDesc[] describeOutput() {
        return new DataDesc[0];
    }

    /** {@inheritDoc} */
    @Override
    public String[] getArtifactNames() {
        return new String[0];
    }

    /** {@inheritDoc} */
    @Override
    public <T> T getArtifact(String name, Function<InputStream, T> function) throws IOException {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public URL getArtifact(String name) throws IOException {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public InputStream getArtifactAsStream(String name) throws IOException {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Model cast(DataType dataType) {
        return null;
    }
}
