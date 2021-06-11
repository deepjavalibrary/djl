import ai.djl.Model;
import ai.djl.modality.Classifications;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Pipeline;
import ai.djl.translate.ServingTranslator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.List;
import java.util.Map;

public class MyTranslator implements ServingTranslator {

    private List<String> classes;

    @Override
    public NDList processInput(TranslatorContext ctx, Input input) throws Exception {
        ctx.setAttachment("input", input);
        byte[] data = input.getContent().valueAt(0);
        ImageFactory factory = ImageFactory.getInstance();
        Image image = factory.fromInputStream(new ByteArrayInputStream(data));

        NDArray array = image.toNDArray(ctx.getNDManager(), Image.Flag.GRAYSCALE);
        Pipeline pipeline = new Pipeline();
        pipeline.add(new CenterCrop());
        pipeline.add(new Resize(28, 28));
        pipeline.add(new ToTensor());
        return pipeline.transform(new NDList(array));
    }

    @Override
    public Output processOutput(TranslatorContext ctx, NDList list) {
        NDArray probabilitiesNd = list.singletonOrThrow();
        probabilitiesNd = probabilitiesNd.softmax(0);
        Classifications classifications = new Classifications(classes, probabilitiesNd);

        Input input = (Input) ctx.getAttachment("input");
        Output output = new Output(input.getRequestId(), 200, "OK");
        output.setContent(classifications.toJson());
        return output;
    }

    @Override
    public void setArguments(Map<String, ?> arguments) {
    }

    @Override
    public void prepare(NDManager manager, Model model) throws IOException {
        if (classes == null) {
            classes = model.getArtifact("synset.txt", Utils::readLines);
        }
    }

    @Override
    public Batchifier getBatchifier() {
        return Batchifier.STACK;
    }
}

