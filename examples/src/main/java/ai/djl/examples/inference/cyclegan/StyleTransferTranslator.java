package ai.djl.examples.inference.cyclegan;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class StyleTransferTranslator implements Translator<Image, Image> {

    private static final Logger logger = LoggerFactory.getLogger(StyleTransferTranslator.class);
    private static double time;

    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {

        NDManager manager = ctx.getNDManager();
        NDArray image = NDArrays.stack(input.toNDArray(manager).split(3, 2));
        NDList list = new NDList(image.squeeze().expandDims(0).toType(DataType.FLOAT32, false));
        time = System.currentTimeMillis();
        return list;
    }

    @Override
    public Image processOutput(TranslatorContext ctx, NDList list) {
        time = System.currentTimeMillis() - time;
        logger.info("time: {} milliseconds", time);

        NDArray output = list.get(0).addi(1).muli(128).toType(DataType.UINT8, false);
        Image img = ImageFactory.getInstance().fromNDArray(output.squeeze());

        return img;
    }

    @Override
    public Batchifier getBatchifier() {
        return null;
    }
}
