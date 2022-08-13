package ai.djl.gluonTS.transform.field;

import ai.djl.gluonTS.GluonTSData;
import ai.djl.gluonTS.transform.GluonTSTransform;
import ai.djl.ndarray.NDManager;

import java.util.List;

public class selectField implements GluonTSTransform {

    private final List<String> inputFields;

    public selectField(List<String> inputFields) {
        this.inputFields = inputFields;
    }

    @Override
    public GluonTSData transform(NDManager manager, GluonTSData data) {
        return Field.selectField(
            manager,
            inputFields,
            data
        );
    }
}
