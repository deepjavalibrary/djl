package ai.djl.gluonTS.transform.field;

import ai.djl.gluonTS.GluonTSData;
import ai.djl.gluonTS.dataset.FieldName;
import ai.djl.gluonTS.transform.GluonTSTransform;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

public class SetField implements GluonTSTransform {

    private final FieldName outputField;
    private final NDArray value;

    public SetField(FieldName outputField, NDArray value) {
        this.outputField = outputField;
        this.value = value;
    }

    @Override
    public GluonTSData transform(NDManager manager, GluonTSData data) {
        NDArray _value = manager.create(value.getShape());
        value.copyTo(_value);
        return Field.setField(manager, outputField, _value, data);
    }
}
