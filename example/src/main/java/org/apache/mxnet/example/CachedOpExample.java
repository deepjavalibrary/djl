package org.apache.mxnet.example;

import com.amazon.ai.example.util.AbstractExample;
import com.amazon.ai.example.util.LogUtils;
import com.amazon.ai.image.Images;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.Shape;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.FloatBuffer;
import org.apache.mxnet.engine.CachedOp;
import org.apache.mxnet.engine.MxModel;
import org.apache.mxnet.engine.MxNDFactory;
import org.slf4j.Logger;

public final class CachedOpExample extends AbstractExample {

    private static final Logger logger = LogUtils.getLogger(CachedOpExample.class);

    private CachedOpExample() {}

    public static void main(String[] args) {
        new CachedOpExample().runExample(args);
    }

    @Override
    public void predict(String modelDir, String modelName, BufferedImage img, int iteration) {
        String modelPathPrefix = modelDir + '/' + modelName;
        // Pre processing
        int topK = 5;
        BufferedImage image = Images.reshapeImage(img, 224, 224);
        FloatBuffer data = Images.toFloatBuffer(image);
        try (MxNDFactory factory = MxNDFactory.SYSTEM_FACTORY.newSubFactory()) {
            NDArray sfLabel = factory.create(new DataDesc(new Shape(1)));
            NDArray nd = factory.create(new DataDesc(new Shape(1, 3, 224, 224)));
            nd.set(data);

            // Inference Logic
            CachedOp op = CachedOp.loadModel(factory, modelPathPrefix, 0);
            NDList result = op.forward(new NDList(nd, sfLabel));
            // Post Processing
            NDArray sorted = result.get(0).argsort(-1, false);
            float[] top = sorted.slice(0, 1).toFloatArray();
            File synsetFile = new File(new File(modelPathPrefix).getParentFile(), "synset.txt");
            String[] synset = MxModel.loadSynset(synsetFile);
            for (int i = 0; i < topK; i++) {
                String className = synset[(int) top[i]];
                logger.info(String.format("Class %d: %s", i, className));
            }
        }
    }
}
