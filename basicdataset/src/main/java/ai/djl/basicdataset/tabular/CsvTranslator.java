/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
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
import java.util.List;

/** A {@link Translator} that converts between CSV text and {@link NDList}. */
public class CsvTranslator implements Translator<String, String> {

    private final CSVFormat csvFormat;

    /** Constructs a CsvTranslator. */
    public CsvTranslator() {
        this.csvFormat = CSVFormat.INFORMIX_UNLOAD_CSV;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, String csvData) throws TranslateException {
        StringReader reader = new StringReader(csvData);

        try (CSVParser parser = csvFormat.parse(reader)) {
            List<CSVRecord> records = parser.getRecords();
            if (records.isEmpty()) {
                throw new TranslateException("CSV data is empty");
            }

            int rowStart = 0;

            // Skip header if present
            if (isHeaderRow(records.get(0))) {
                rowStart = 1;
            }

            int numRows = records.size() - rowStart;
            int expectedCols =
                    records.get(rowStart).size(); // assume first data row sets column count

            float[][] data = new float[numRows][expectedCols];

            for (int i = 0; i < numRows; i++) {
                CSVRecord record = records.get(i + rowStart);

                if (record.size() != expectedCols) {
                    throw new TranslateException(
                            "Row "
                                    + record.getRecordNumber()
                                    + " has "
                                    + record.size()
                                    + " columns, expected "
                                    + expectedCols);
                }

                for (int j = 0; j < expectedCols; j++) {
                    data[i][j] = parseFloat(record.get(j), record.getRecordNumber(), j);
                }
            }

            NDManager manager = ctx.getNDManager();
            NDArray ndArray = manager.create(data);
            return new NDList(ndArray);
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
        int len = str.length();
        for (int i = 0; i < len; i++) {
            char c = str.charAt(i);
            if ((c < '0' || c > '9') && c != '-' && c != '.' && c != 'e' && c != 'E' && c != '+') {
                return false;
            }
        }
        return true;
    }

    private float parseFloat(String value, long row, int col) throws TranslateException {
        if (value == null || value.isEmpty()) {
            return Float.NaN;
        }

        int len = value.length();
        if (value.charAt(0) <= ' ' || value.charAt(len - 1) <= ' ') {
            value = value.trim();
            if (value.isEmpty()) {
                return Float.NaN;
            }
        }

        try {
            return Float.parseFloat(value);
        } catch (NumberFormatException e) {
            throw new TranslateException(
                    "Non-numeric value '" + value + "' at row " + row + ", column " + col, e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public String processOutput(TranslatorContext ctx, NDList list) throws TranslateException {
        NDArray array = list.singletonOrThrow();

        try (StringWriter writer = new StringWriter();
                CSVPrinter printer = new CSVPrinter(writer, csvFormat)) {
            float[] data = array.toType(DataType.FLOAT32, false).toFloatArray();
            long[] shape = array.getShape().getShape();

            if (shape.length == 1) {
                printRow(printer, data, 0, data.length);
            } else if (shape.length == 2) {
                int rows = (int) shape[0];
                int cols = (int) shape[1];
                for (int i = 0; i < rows; i++) {
                    printRow(printer, data, i * cols, cols);
                }
            } else {
                throw new TranslateException(
                        "Only 1D or 2D arrays supported, found shape: " + array.getShape());
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
