package ai.djl.gluonTS.transform.convert;

import ai.djl.gluonTS.GluonTSData;
import ai.djl.gluonTS.dataset.FieldName;
import ai.djl.gluonTS.transform.GluonTSTransform;
import ai.djl.ndarray.NDManager;

import java.util.List;

public class VstackFeatures implements GluonTSTransform {

    private FieldName outputField;
    private List<FieldName> inputFields;
    private boolean dropInputs;
    private boolean hStack;

    public VstackFeatures(
            FieldName outputField,
            List<FieldName> inputFields,
            boolean dropInputs,
            boolean hStack) {
        this.outputField = outputField;
        this.inputFields = inputFields;
        this.dropInputs = dropInputs;
        this.hStack = hStack;
    }

    public VstackFeatures(FieldName outputField, List<FieldName> inputFields) {
        this(outputField, inputFields, true, false);
    }

    /** {@inheritDoc} */
    @Override
    public GluonTSData transform(NDManager manager, GluonTSData data) {
        return Convert.vstackFeatures(manager, outputField, inputFields, dropInputs, hStack, data);
    }
}
