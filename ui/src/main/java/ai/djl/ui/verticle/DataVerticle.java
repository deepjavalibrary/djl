package ai.djl.ui.verticle;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.*;
import java.util.stream.Collectors;

import ai.djl.Device;
import ai.djl.metric.Metrics;
import ai.djl.training.Trainer;
import ai.djl.training.listener.EvaluatorTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.ui.data.MetricInfo;
import ai.djl.ui.data.ModelInfo;
import ai.djl.ui.data.TrainerInfo;
import io.vavr.control.Try;
import io.vertx.core.AbstractVerticle;
import io.vertx.core.json.Json;
import io.vertx.core.logging.Logger;
import io.vertx.core.logging.LoggerFactory;

public class DataVerticle extends AbstractVerticle {

    private final Logger LOGGER = LoggerFactory.getLogger(WebVerticle.class.getCanonicalName());

    public static final String ADDRESS_TRAINER_REQUEST = "trainer-request";
    public static final String ADDRESS_TRAINER = "trainer";

    private Trainer trainer;
    private int currentEpoch = 1;
    private TrainerInfo.State currentState = TrainerInfo.State.Undefined;
    private int trainingProgress = 0;
    private int validatingProgress = 0;
    private int batchSize = 0;
    private final Map<String, List<MetricInfo>> performance = new HashMap<>();
    private long updateInterval = 500;
    private long lastUpdate = System.currentTimeMillis();

    @Override
    public void start() throws Exception {
        LOGGER.info("DataVerticle starting...");
        vertx.eventBus().consumer(ADDRESS_TRAINER_REQUEST, event -> Try.run(() -> sendTrainer()));
    }

    public void setEpoch(Trainer trainer) {
        this.trainer = trainer;
        this.validatingProgress = 0;
        this.trainingProgress = 0;
        this.performance.clear();
        this.currentEpoch++;
        sendTrainer();
    }

    public void setTrainingBatch(Trainer trainer, TrainingListener.BatchData batchData) {
        setBatch(trainer, batchData, TrainerInfo.State.Training);
    }

    public void setValidationBatch(Trainer trainer, TrainingListener.BatchData batchData) {
        setBatch(trainer, batchData, TrainerInfo.State.Training);
    }

    public void sendTrainer() {
        Try.run(() -> {
            ModelInfo mi = ModelInfo.builder()
                    .name(Try.of(() -> trainer.getModel().getName()).getOrElse("Noname"))
                    .block(Try.of(() -> trainer.getModel().getBlock().toString()).getOrElse("Undefined"))
                    .build();

            TrainerInfo trainerInfo = TrainerInfo.builder()
                    .devices(getDevices())
                    .modelInfo(mi)
                    .state(currentState)
                    .epoch(currentEpoch)
                    .speed(getSpeed())
                    .trainingProgress(trainingProgress)
                    .validatingProgress(validatingProgress)
                    .metrics(Map.copyOf(performance))
                    .metricNames(new ArrayList<>(performance.keySet()))
                    .metricsSize(performance.isEmpty() ? 0 : performance.values().iterator().next().size())
                    .build();
            vertx.eventBus().publish(ADDRESS_TRAINER, Json.encode(trainerInfo));
        }).onFailure(throwable -> LOGGER.error("", throwable));
    }

    private void setBatch(Trainer trainer, TrainingListener.BatchData batchData, TrainerInfo.State state) {
        this.trainer = trainer;
        this.batchSize = batchData.getBatch().getSize();
        this.currentState = state;
        this.trainingProgress = (int) ((batchData.getBatch().getProgress() + 1) * 100 / batchData.getBatch().getProgressTotal());
        setPerformance();
        if (isTimeToUpdate()) {
            sendTrainer();
            lastUpdate = System.currentTimeMillis();
        }
    }

    private void setPerformance() {
        Try.run(() -> {
            Metrics metrics = trainer.getMetrics();
            trainer.getEvaluators().forEach(e -> {
                String metricName = EvaluatorTrainingListener.metricName(e, EvaluatorTrainingListener.TRAIN_PROGRESS);
                if (metrics.hasMetric(metricName)) {
                    List<MetricInfo> mis = getMetrics(e.getName());
                    mis.add(MetricInfo.builder().name(e.getName()).x(mis.size()).y(metrics.latestMetric(metricName).getValue().floatValue()).build());
                    setMetrics(e.getName(), mis);
                }
            });
        }).onFailure(throwable -> LOGGER.error("", throwable));
    }

    private void setMetrics(String name, List<MetricInfo> metricInfos) {
        performance.put(name, metricInfos);
    }

    private List<MetricInfo> getMetrics(String name) {
        return performance.containsKey(name) ? performance.get(name) : new ArrayList();
    }

    private boolean isTimeToUpdate() {
        return System.currentTimeMillis() - lastUpdate > updateInterval;
    }

    private List<String> getDevices() {
        return Try.of(() -> trainer.getDevices().stream().map(device -> device.toString()).collect(Collectors.toList()))
                .getOrElse(Arrays.asList(Device.cpu().toString()));
    }

    private BigDecimal getSpeed() {
        return Try.of(() -> {
            Metrics metrics = trainer.getMetrics();
            if (metrics != null && metrics.hasMetric("train")) {
                float batchTime = metrics.latestMetric("train").getValue().longValue() / 1_000_000_000f;
                return BigDecimal.valueOf(batchSize / batchTime).setScale(2, RoundingMode.HALF_UP);
            }
            return BigDecimal.ZERO;
        }).getOrElse(BigDecimal.ZERO);

    }
}
