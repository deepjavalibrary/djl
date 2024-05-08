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

import ai.djl.basicdataset.tabular.TabularDataset;
import ai.djl.util.Progress;

import tech.tablesaw.api.Row;
import tech.tablesaw.api.Table;
import tech.tablesaw.io.ReadOptions;

import java.util.Collections;
import java.util.List;

/** {@code TablesawDataset} represents the dataset that stored in a .csv file. */
public class TablesawDataset extends TabularDataset {

    protected ReadOptions readOptions;
    protected Table table;

    protected TablesawDataset(TablesawBuilder<?> builder) {
        super(builder);
        readOptions = builder.readOptions;
    }

    /** {@inheritDoc} */
    @Override
    public String getCell(long rowIndex, String featureName) {
        Row row = table.row(Math.toIntExact(rowIndex));
        return row.getString(featureName);
    }

    /** {@inheritDoc} */
    @Override
    protected long availableSize() {
        return table.rowCount();
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(Progress progress) {
        table = Table.read().usingOptions(readOptions);
        prepareFeaturizers();
    }

    /**
     * Creates a builder to build a {@link TablesawDataset}.
     *
     * @return a new builder
     */
    public static TablesawBuilder<?> builder() {
        return new TablesawBuilder<>();
    }

    /**
     * Returns the column names of the Tablesaw file.
     *
     * @return a list of column name
     */
    public List<String> getColumnNames() {
        if (table.isEmpty()) {
            return Collections.emptyList();
        }
        return table.columnNames();
    }

    /** Used to build a {@link TablesawDataset}. */
    public static class TablesawBuilder<T extends TablesawBuilder<T>>
            extends TabularDataset.BaseBuilder<T> {

        protected ReadOptions readOptions;

        /** {@inheritDoc} */
        @Override
        @SuppressWarnings("unchecked")
        protected T self() {
            return (T) this;
        }

        /**
         * Sets the reading options.
         *
         * @param readOptions the {@code ReadOptions}
         * @return this builder
         */
        public T setReadOptions(ReadOptions readOptions) {
            this.readOptions = readOptions;
            return self();
        }

        /**
         * Builds the new {@link TablesawDataset}.
         *
         * @return the new {@link TablesawDataset}
         */
        public TablesawDataset build() {
            return new TablesawDataset(this);
        }
    }
}
