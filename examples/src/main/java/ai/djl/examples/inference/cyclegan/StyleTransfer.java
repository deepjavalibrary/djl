package ai.djl.examples.inference.cyclegan;

import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class StyleTransfer {

    private static final Logger logger = LoggerFactory.getLogger(StyleTransfer.class);

    private StyleTransfer() {}

    private enum Artist {
        CEZANNE,
        MONET,
        UKIYOE,
        VANGOGH
    }

    public static void main(String[] args) throws Exception {
        Artist artist = Artist.VANGOGH;

        Image input = ImageFactory.getInstance().fromFile(Paths.get("src/main/resources/mtn.png"));
        Image output = transfer(input, artist);

        if (output == null) {
            logger.info("This example only works for PyTorch Engine");
        } else {
            logger.info("Using PyTorch Engine. " + artist + " painting generated.");
            save(output, artist.toString(), "build/output/cyclegan/");
        }
    }

    public static Image transfer(Image image, Artist artist) throws Exception {

        if (!"PyTorch".equals(Engine.getInstance().getEngineName())) {
            return null;
        }

        String modelName = "style_" + artist;
        String modelPath = "src/main/resources/models/";

        Criteria<Image, Image> criteria =
                Criteria.builder()
                        .setTypes(Image.class, Image.class)
                        .optModelPath(Paths.get(modelPath))
                        .optModelName(modelName)
                        .optTranslator(new StyleTransferTranslator())
                        .build();

        try (ZooModel<Image, Image> model = criteria.loadModel();
                Predictor<Image, Image> styler = model.newPredictor()) {
            return styler.predict(image);
        }
    }

    public static void save(Image image, String name, String path) throws IOException {
        Path outputPath = Paths.get(path);
        Files.createDirectories(outputPath);
        Path imagePath = outputPath.resolve(name + ".png");
        image.save(Files.newOutputStream(imagePath), "png");
    }
}
