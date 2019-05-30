package org.apache.mxnet.example;

import com.amazon.ai.Model;
import com.amazon.ai.example.util.AbstractExample;
import com.amazon.ai.example.util.LogUtils;
import com.amazon.ai.image.Images;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.Shape;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.apache.mxnet.engine.CachedOp;
import org.apache.mxnet.engine.MxModel;
import org.apache.mxnet.engine.MxNDFactory;
import org.apache.mxnet.jna.JnaUtils;
import org.slf4j.Logger;

public final class CachedOpExample extends AbstractExample {

    private static final Logger logger = LogUtils.getLogger(CachedOpExample.class);

    private CachedOpExample() {}

    public static void main(String[] args) {
        new CachedOpExample().runExample(args);
    }

    @Override
    public void predict(String modelDir, String modelName, BufferedImage img, int iteration)
            throws IOException {
        String modelPathPrefix = modelDir + '/' + modelName;

        Model model = Model.loadModel(modelPathPrefix, 0);

        // Pre processing
        int topK = 5;
        BufferedImage image = Images.reshapeImage(img, 224, 224);
        FloatBuffer data = Images.toFloatBuffer(image);
        try (MxNDFactory factory = MxNDFactory.SYSTEM_FACTORY.newSubFactory()) {
            NDArray sfLabel = factory.create(new DataDesc(new Shape(1)));
            NDArray nd = factory.create(new DataDesc(new Shape(1, 3, 224, 224)));
            nd.set(data);

            // Inference Logic
            long init = System.nanoTime();
            CachedOp op = JnaUtils.createCachedOp((MxModel) model, factory);
            long loadModel = System.nanoTime();
            logger.info(String.format("bind model  = %.3f ms.", (loadModel - init) / 1000000f));
            List<Long> inferenceTime = new ArrayList<>(iteration);
            long firstInfStart = System.nanoTime();
            NDList result = op.forward(new NDList(nd, sfLabel));
            result.get(0).waitToRead();
            long firstInfEnd = System.nanoTime();
            logger.info("First Inference: " + (firstInfEnd - firstInfStart) / 1000000f + " ms");
            for (int i = 0; i < iteration; ++i) {
                long begin = System.nanoTime();
                result = op.forward(new NDList(nd, sfLabel));
                result.get(0).waitToRead();
                long inference = System.nanoTime();
                inferenceTime.add(inference - begin);
                logger.info("Time cost: " + (inference - begin) / 1000000f + " ms");
            }
            Collections.sort(inferenceTime);

            float p50 = inferenceTime.get(iteration / 2) / 1000000f;
            float p90 = inferenceTime.get(iteration * 9 / 10) / 1000000f;
            float p99 = inferenceTime.get(iteration * 99 / 100) / 1000000f;

            logger.info(
                    String.format(
                            "inference P50: %.3f ms, P90: %.3f ms, P99 %.3f ms", p50, p90, p99));
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
