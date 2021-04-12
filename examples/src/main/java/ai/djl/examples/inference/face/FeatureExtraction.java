package ai.djl.examples.inference.face;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.*;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class FeatureExtraction {

    private static final Logger logger = LoggerFactory.getLogger(FeatureExtraction.class);

    private FeatureExtraction() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        Path imageFile = Paths.get("src/test/resources/kana1.jpg");
        Image img = ImageFactory.getInstance().fromFile(imageFile);

        float[] feature = FeatureExtraction.predict(img);
        if (feature == null) {
            logger.info("Failed to extract feature.");
        } else {
            logger.info(Arrays.toString(feature));
        }
    }

    public static float[] predict(Image img)
            throws IOException, ModelException, TranslateException {
        img.getWrappedImage();
        Criteria<Image, float[]> criteria =
                Criteria.builder()
                        .setTypes(Image.class, float[].class)
                        .optModelUrls(
                                "https://djl-model.oss-cn-hongkong.aliyuncs.com/face_feature.zip")
                        // Load model from local file, e.g:
                        // "file:///Users/calvin/pytorch_models/face_feature/"
                        // .optModelUrls("file:///path/to/model_dir/")
                        .optModelName("face_feature") // specify model file prefix
                        .optTranslator(new FaceFeatureTranslator())
                        .optProgress(new ProgressBar())
                        .optEngine("PyTorch") // Use PyTorch engine
                        .build();

        ZooModel model = ModelZoo.loadModel(criteria);
        Predictor<Image, float[]> predictor = model.newPredictor();
        float[] feature = predictor.predict(img);
        return feature;
    }

    private static final class FaceFeatureTranslator implements Translator<Image, float[]> {
        protected Batchifier batchifier = Batchifier.STACK;

        FaceFeatureTranslator() {}

        @Override
        public NDList processInput(TranslatorContext ctx, Image input) {
            NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
            Pipeline pipeline = new Pipeline();
            pipeline
                    // .add(new Resize(160))
                    .add(new ToTensor())
                    .add(
                            new Normalize(
                                    new float[] {127.5f / 255.0f, 127.5f / 255.0f, 127.5f / 255.0f},
                                    new float[] {
                                        128.0f / 255.0f, 128.0f / 255.0f, 128.0f / 255.0f
                                    }));

            NDList list = pipeline.transform(new NDList(array));
            return list;
        }

        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            NDList result = new NDList();
            long numOutputs = list.singletonOrThrow().getShape().get(0);
            for (int i = 0; i < numOutputs; i++) {
                result.add(list.singletonOrThrow().get(i));
            }
            float[][] embeddings =
                    result.stream().map(NDArray::toFloatArray).toArray(float[][]::new);
            float[] feature = new float[embeddings.length];
            for (int i = 0; i < embeddings.length; i++) {
                feature[i] = embeddings[i][0];
            }
            return feature;
        }

        @Override
        public Batchifier getBatchifier() {
            return batchifier;
        }
    }
}
