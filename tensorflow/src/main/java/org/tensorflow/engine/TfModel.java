package org.tensorflow.engine;

import com.amazon.ai.Model;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.DataType;
import com.amazon.ai.ndarray.types.Shape;
import com.google.protobuf.InvalidProtocolBufferException;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.function.Function;
import org.tensorflow.Graph;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;
import org.tensorflow.framework.TensorInfo;
import org.tensorflow.framework.TensorShapeProto;

public class TfModel implements Model {

    private Path modelDir;
    private SavedModelBundle bundle;
    private DataDesc[] inputDesc;
    private DataDesc[] outputDesc;

    TfModel(Path modelDir, SavedModelBundle bundle) throws InvalidProtocolBufferException {
        this.modelDir = modelDir;
        this.bundle = bundle;
        SignatureDef sig =
                MetaGraphDef.parseFrom(this.bundle.metaGraphDef())
                        .getSignatureDefOrThrow("serving_default");
        inputDesc = constructDataDescFromModel(sig.getInputsMap());
        outputDesc = constructDataDescFromModel(sig.getOutputsMap());
    }

    private DataDesc[] constructDataDescFromModel(Map<String, TensorInfo> info) {
        DataDesc[] descs = new DataDesc[info.size()];
        int dataDescIter = 0;
        for (Map.Entry<String, TensorInfo> entry : info.entrySet()) {
            TensorInfo t = entry.getValue();
            // StringBuilder layout = new StringBuilder();
            int[] shape = new int[t.getTensorShape().getDimCount()];
            int dimIter = 0;
            for (TensorShapeProto.Dim dim : t.getTensorShape().getDimList()) {
                // layout.append(dim.getName());
                shape[dimIter] = (int) dim.getSize();
                dimIter++;
            }
            // TODO: Add DataType mapping from framework.DataType
            // TODO: Add Layout mapping for the layout
            descs[dataDescIter] = new DataDesc(new Shape(shape), null, t.getName(), null);
            dataDescIter++;
        }
        return descs;
    }

    public static TfModel loadModel(String modelDir, String... tags)
            throws InvalidProtocolBufferException {
        if (tags == null || tags.length == 0) {
            tags = new String[] {"serve"};
        }
        return new TfModel(Paths.get(modelDir), SavedModelBundle.load(modelDir, tags));
    }

    public static TfModel loadModel(
            String modelDir, byte[] configProto, byte[] runOptions, String... tags)
            throws InvalidProtocolBufferException {
        SavedModelBundle bundle =
                SavedModelBundle.loader(modelDir)
                        .withConfigProto(configProto)
                        .withRunOptions(runOptions)
                        .withTags(tags)
                        .load();
        return new TfModel(Paths.get(modelDir), bundle);
    }

    public Graph getGraph() {
        return bundle.graph();
    }

    public Session getSession() {
        return bundle.session();
    }

    private byte[] getMetaGraphDef() {
        return bundle.metaGraphDef();
    }

    /** {@inheritDoc} */
    @Override
    public DataDesc[] describeInput() {
        return inputDesc;
    }

    /** {@inheritDoc} */
    @Override
    public DataDesc[] describeOutput() {
        return outputDesc;
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
    public URL getArtifact(String artifactName) throws IOException {
        if (artifactName == null) {
            throw new IllegalArgumentException("artifactName cannot be null");
        }
        Path file = modelDir.resolve(artifactName);
        if (Files.exists(file) && Files.isReadable(file)) {
            return file.toUri().toURL();
        }
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
