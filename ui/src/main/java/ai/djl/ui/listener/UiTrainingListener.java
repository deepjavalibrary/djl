package ai.djl.ui.listener;

import ai.djl.training.Trainer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.ui.verticle.DataVerticle;
import ai.djl.ui.verticle.WebVerticle;
import io.vertx.core.Vertx;
import io.vertx.core.logging.Logger;
import io.vertx.core.logging.LoggerFactory;

public class UiTrainingListener  implements TrainingListener {

    private final Logger LOGGER = LoggerFactory.getLogger(UiTrainingListener.class.getCanonicalName());

    private final Vertx vertx = Vertx.vertx();
    private final DataVerticle dataVerticle = new DataVerticle();
    private final WebVerticle webVerticle = new WebVerticle();

    public UiTrainingListener() {
        LOGGER.info("UiTrainingListener starting...");
        vertx.deployVerticle(dataVerticle);
        vertx.deployVerticle(webVerticle);
    }

    @Override
    public void onEpoch(Trainer trainer) {
        dataVerticle.setEpoch(trainer);
    }

    @Override
    public void onTrainingBatch(Trainer trainer, BatchData batchData) {
        dataVerticle.setTrainingBatch(trainer, batchData);
    }

    @Override
    public void onValidationBatch(Trainer trainer, BatchData batchData) {
        dataVerticle.setValidationBatch(trainer, batchData);
    }

    @Override
    public void onTrainingBegin(Trainer trainer) {
        LOGGER.info("onTrainingBegin ...");
//        dataVerticle.setTrainer(trainer, numEpochs);
    }

    @Override
    public void onTrainingEnd(Trainer trainer) {
        LOGGER.info("onTrainingEnd ...");
        vertx.close();
    }
}
