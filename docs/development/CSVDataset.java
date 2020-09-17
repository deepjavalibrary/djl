import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.util.Progress;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class CSVDataset extends RandomAccessDataset {

    private final List<CSVRecord> csvRecords;

    private CSVDataset(Builder builder) {
        super(builder);
        csvRecords = builder.csvRecords;
    }

    @Override
    public Record get(NDManager manager, long index) {
        CSVRecord record = csvRecords.get(Math.toIntExact(index));
        NDArray datum = manager.create(encode(record.get("url")));
        NDArray label = manager.create(Float.parseFloat(record.get("isMalicious")));
        return new Record(new NDList(datum), new NDList(label));
    }

    @Override
    public long availableSize() {
        return csvRecords.size();
    }

    // we encode the url String based on the count of the character from a to z.
    private int[] encode(String url) {
        url = url.toLowerCase();
        int[] encoding = new int[26];
        for (char ch : url.toCharArray()) {
            int index = ch - 'a';
            if (index < 26 && index >= 0) {
                encoding[ch - 'a']++;
            }
        }
        return encoding;
    }

    @Override
    public void prepare(Progress progress) {}
    
    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder extends BaseBuilder<Builder> {

        List<CSVRecord> csvRecords;

        Builder(){}

        @Override
        protected Builder self() {
            return this;
        }

        CSVDataset build() throws IOException {
            String csvFilePath = "path/malicious_url_data.csv";
            try (Reader reader = Files.newBufferedReader(Paths.get(csvFilePath));
                 CSVParser csvParser =
                         new CSVParser(
                                 reader,
                                 CSVFormat.DEFAULT
                                         .withHeader("url", "isMalicious")
                                         .withFirstRecordAsHeader()
                                         .withIgnoreHeaderCase()
                                         .withTrim())) {
                csvRecords = csvParser.getRecords();
            }
            return new CSVDataset(this);
        }
    }
}
