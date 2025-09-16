/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.basicdataset.tabular;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** A {@link Translator} that converts between CSV text and {@link NDList}. */
public class CsvTranslator implements Translator<String, String> {

    private final CSVFormat csvFormat;

    /**
     * Constructs a CsvTranslator.
     *
     * @param arguments the arguments (unused but required for reflection)
     */
    @SuppressWarnings({"PMD.UnusedFormalParameter", "deprecation"})
    public CsvTranslator(Map<String, ?> arguments) {
        this.csvFormat = CSVFormat.newFormat(',').withRecordSeparator("\n");
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, String csvData) throws TranslateException {
        try (CSVParser parser = csvFormat.parse(new StringReader(csvData))) {
            List<float[]> rows = new ArrayList<>();
            int expectedCols = -1;
            boolean headerSkipped = false;

            for (CSVRecord record : parser) {
                if (!headerSkipped && isHeaderRow(record)) {
                    headerSkipped = true;
                    continue;
                }

                if (expectedCols == -1) {
                    expectedCols = record.size();
                } else if (record.size() != expectedCols) {
                    throw new TranslateException(
                            String.format(
                                    "Row %d has %d columns, expected %d",
                                    record.getRecordNumber(), record.size(), expectedCols));
                }

                float[] row = new float[expectedCols];
                for (int j = 0; j < expectedCols; j++) {
                    row[j] = parseFloat(record.get(j), record.getRecordNumber(), j);
                }
                rows.add(row);
            }

            if (rows.isEmpty()) {
                throw new TranslateException("CSV data is empty");
            }

            float[][] data = rows.toArray(new float[0][]);
            return new NDList(ctx.getNDManager().create(data));
        } catch (IOException e) {
            throw new TranslateException("Failed to process CSV input", e);
        }
    }

    private boolean isHeaderRow(CSVRecord record) {
        for (String cell : record) {
            if (isNumeric(cell.trim())) {
                return false;
            }
        }
        return true;
    }

    private boolean isNumeric(String str) {
        if (str == null || str.isEmpty()) {
            return false;
        }
        try {
            Float.parseFloat(str);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    private float parseFloat(String value, long row, int col) throws TranslateException {
        try {
            return Float.parseFloat(value.trim());
        } catch (NumberFormatException e) {
            throw new TranslateException(
                    String.format("Non-numeric value '%s' at row %d, column %d", value, row, col),
                    e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public String processOutput(TranslatorContext ctx, NDList list) throws TranslateException {
        try (StringWriter writer = new StringWriter();
                CSVPrinter printer = new CSVPrinter(writer, csvFormat)) {

            for (NDArray array : list) {
                float[] data =
                        array.toType(ai.djl.ndarray.types.DataType.FLOAT32, false).toFloatArray();
                long[] shape = array.getShape().getShape();

                if (shape.length == 1) {
                    // 1D array → single row
                    printRow(printer, data, 0, data.length);
                } else if (shape.length == 2) {
                    int rows = (int) shape[0];
                    int cols = (int) shape[1];
                    for (int i = 0; i < rows; i++) {
                        printRow(printer, data, i * cols, cols);
                    }
                } else {
                    throw new TranslateException(
                            "Only 1D or 2D arrays can be converted to CSV, found shape: "
                                    + array.getShape());
                }
            }

            return writer.toString();
        } catch (IOException e) {
            throw new TranslateException("Failed to generate CSV output", e);
        }
    }

    private void printRow(CSVPrinter printer, float[] data, int offset, int length)
            throws IOException {
        for (int i = 0; i < length; i++) {
            printer.print(data[offset + i]);
        }
        printer.println();
    }
}
