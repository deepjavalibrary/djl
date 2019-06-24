package org.tensorflow.engine;

import com.amazon.ai.Model;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.DataType;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.function.Function;

public class TfModel implements Model {

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
