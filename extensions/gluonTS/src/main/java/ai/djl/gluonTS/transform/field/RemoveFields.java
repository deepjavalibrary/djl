package ai.djl.gluonTS.transform.field;

import ai.djl.gluonTS.GluonTSData;
import ai.djl.gluonTS.dataset.FieldName;
import ai.djl.gluonTS.transform.GluonTSTransform;
import ai.djl.ndarray.NDManager;

import java.util.List;

public class RemoveFields implements GluonTSTransform {

    private final List<FieldName> fieldNames;

    public RemoveFields(List<FieldName> fieldNames) {
        this.fieldNames = fieldNames;
    }

    /** {@inheritDoc} */
    @Override
    public GluonTSData transform(NDManager manager, GluonTSData data) {
        return Field.removeFields(manager, fieldNames, data);
    }
}
