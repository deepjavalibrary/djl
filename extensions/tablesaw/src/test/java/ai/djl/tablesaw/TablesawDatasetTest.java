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
package ai.djl.tablesaw;

import ai.djl.Model;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Blocks;
import ai.djl.training.dataset.Record;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import tech.tablesaw.api.ColumnType;
import tech.tablesaw.io.csv.CsvReadOptions;

import java.io.IOException;
import java.net.URL;

public class TablesawDatasetTest {

    @Test
    public void testTablesawDataset() throws IOException, TranslateException {
        URL dailyDelhiClimateTest =
                new URL(
                        "https://mlrepo.djl.ai/dataset/tabular/ai/djl/basicdataset/daily-delhi-climate/3.0/DailyDelhiClimateTest.csv");

        try (Model model = Model.newInstance("model")) {
            model.setBlock(Blocks.identityBlock());

            NDManager manager = model.getNDManager();
            TablesawDataset dailyDelhiClimate =
                    TablesawDataset.builder()
                            .setReadOptions(
                                    CsvReadOptions.builder(dailyDelhiClimateTest)
                                            .header(true)
                                            .columnTypes(
                                                    new ColumnType[] {
                                                        ColumnType.STRING,
                                                        ColumnType.STRING,
                                                        ColumnType.STRING,
                                                        ColumnType.STRING,
                                                        ColumnType.STRING
                                                    })
                                            .build())
                            .addNumericFeature("meantemp")
                            .addNumericFeature("meanpressure")
                            .addNumericLabel("humidity")
                            .setSampling(32, true)
                            .build();
            dailyDelhiClimate.prepare();

            long size = dailyDelhiClimate.size();
            Assert.assertEquals(size, 114);

            Record record = dailyDelhiClimate.get(manager, 3);
            NDList data = record.getData();

            NDList labels = record.getLabels();
            Assert.assertEquals(data.head().toFloatArray(), new float[] {18.7f, 1015.7f});
            Assert.assertEquals(labels.head().toFloatArray(), new float[] {70.05f});
        }
    }
}
