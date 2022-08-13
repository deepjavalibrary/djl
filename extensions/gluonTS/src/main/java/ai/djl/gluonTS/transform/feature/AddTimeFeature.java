package ai.djl.gluonTS.transform.feature;

import ai.djl.gluonTS.GluonTSData;
import ai.djl.gluonTS.dataset.FieldName;
import ai.djl.gluonTS.transform.GluonTSTransform;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

import java.time.LocalDateTime;
import java.util.List;
import java.util.function.BiFunction;

public class AddTimeFeature implements GluonTSTransform {

    private FieldName startField;
    private FieldName targetField;
    private FieldName outputField;
    private List<BiFunction<NDManager, List<LocalDateTime>, NDArray>> timeFeatures;
    private int predictionLength;
    String freq;

    public AddTimeFeature(
            FieldName startField,
            FieldName targetField,
            FieldName outputField,
            List<BiFunction<NDManager, List<LocalDateTime>, NDArray>> timeFeatures,
            int predictionLength,
            String freq) {
        this.startField = startField;
        this.targetField = targetField;
        this.outputField = outputField;
        this.timeFeatures = timeFeatures;
        this.predictionLength = predictionLength;
        this.freq = freq;
    }

    @Override
    public GluonTSData transform(NDManager manager, GluonTSData data) {
        return Feature.addTimeFeature(
                manager,
                startField,
                targetField,
                outputField,
                timeFeatures,
                predictionLength,
                freq,
                data);
    }
}
