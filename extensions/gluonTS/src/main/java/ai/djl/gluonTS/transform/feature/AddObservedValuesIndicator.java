package ai.djl.gluonTS.transform.feature;

import ai.djl.gluonTS.GluonTSData;
import ai.djl.gluonTS.dataset.FieldName;
import ai.djl.gluonTS.transform.GluonTSTransform;
import ai.djl.ndarray.NDManager;

public class AddObservedValuesIndicator implements GluonTSTransform {

    private FieldName targetField;
    private FieldName outputField;

    public AddObservedValuesIndicator(FieldName targetField, FieldName outputField) {
        this.targetField = targetField;
        this.outputField = outputField;
    }

    /** {@inheritDoc} */
    @Override
    public GluonTSData transform(NDManager manager, GluonTSData data) {
        return Feature.addObservedValuesIndicator(manager, targetField, outputField, data);
    }
}
