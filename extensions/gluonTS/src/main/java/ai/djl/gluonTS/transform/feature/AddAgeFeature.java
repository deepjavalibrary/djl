package ai.djl.gluonTS.transform.feature;

import ai.djl.gluonTS.GluonTSData;
import ai.djl.gluonTS.dataset.FieldName;
import ai.djl.gluonTS.transform.GluonTSTransform;
import ai.djl.ndarray.NDManager;

public class AddAgeFeature implements GluonTSTransform {

    private FieldName targetField;
    private FieldName outputField;
    private int predictionLength;
    private boolean logScale;

    public AddAgeFeature(FieldName targetField, FieldName outputField, int predictionLength) {
        this(targetField, outputField, predictionLength, true);
    }

    public AddAgeFeature(
            FieldName targetField, FieldName outputField, int predictionLength, boolean logScale) {
        this.targetField = targetField;
        this.outputField = outputField;
        this.predictionLength = predictionLength;
        this.logScale = logScale;
    }

    /** {@inheritDoc} */
    @Override
    public GluonTSData transform(NDManager manager, GluonTSData data) {
        return Feature.addAgeFeature(
                manager, targetField, outputField, predictionLength, logScale, data);
    }
}
