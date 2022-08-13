package ai.djl.gluonTS.transform.split;

import ai.djl.gluonTS.GluonTSData;
import ai.djl.gluonTS.dataset.FieldName;
import ai.djl.gluonTS.transform.GluonTSTransform;
import ai.djl.gluonTS.transform.InstanceSampler;
import ai.djl.ndarray.NDManager;

import java.util.List;

public class InstanceSplit implements GluonTSTransform {

    private FieldName targetField;
    private FieldName isPadField;
    private FieldName startField;
    private FieldName forecastStartField;
    private InstanceSampler instanceSampler;
    private int pastLength;
    private int futureLength;
    private int leadTime;
    private boolean outputNTC;
    private List<FieldName> timeSeriesFields;
    private float dummyValue;

    public InstanceSplit(FieldName targetField, FieldName isPadField, FieldName startField, FieldName forecastStartField, InstanceSampler instanceSampler, int pastLength, int futureLength, int leadTime, boolean outputNTC, List<FieldName> timeSeriesFields, float dummyValue) {
        this.targetField = targetField;
        this.isPadField = isPadField;
        this.startField = startField;
        this.forecastStartField = forecastStartField;
        this.instanceSampler = instanceSampler;
        this.pastLength = pastLength;
        this.futureLength = futureLength;
        this.leadTime = leadTime;
        this.outputNTC = outputNTC;
        this.timeSeriesFields = timeSeriesFields;
        this.dummyValue = dummyValue;
    }

    public InstanceSplit(FieldName targetField, FieldName isPadField, FieldName startField, FieldName forecastStartField, InstanceSampler instanceSampler, int pastLength, int futureLength, List<FieldName> timeSeriesFields, float dummyValue) {
        this(targetField, isPadField, startField, forecastStartField, instanceSampler, pastLength, futureLength,
            0, true, timeSeriesFields, dummyValue);
    }

    @Override
    public GluonTSData transform(NDManager manager, GluonTSData data) {
        return Split.instanceSplit(
            manager,
            targetField,
            isPadField,
            startField,
            forecastStartField,
            instanceSampler,
            pastLength,
            futureLength,
            leadTime,
            outputNTC,
            timeSeriesFields,
            dummyValue,
            data
        );
    }
}
