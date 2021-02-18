/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving.loading;

import ai.djl.Application;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.Criteria.Builder;
import java.util.Map;

/**
 * creates a criteria object to lookup for model. inherited class defines where the parameters are
 * defined
 *
 * @author erik.bamberg@web.de
 */
public abstract class ModelCriteriaParser<T> {

    /**
     * parse the input to the intermediate abstract data structure used to build criteria from.
     *
     * @param input the input to parse
     * @return intermediate abstract data structure
     */
    protected abstract Parameters parseInput(T input);

    /**
     * parse input and create modelZoo criteria representing the parameters.
     *
     * @param input the input object to parse parameters of.
     * @return criteria the modelzoo criteria which can be used to lookup the requested model.
     */
    public Criteria<?, ?> of(T input) {
        Parameters params = parseInput(input);
        Builder<?, ?> criteriaBuilder =
                Criteria.builder().setTypes(params.inputType, params.outputType);
        if (params.modelUrl != null) {
            criteriaBuilder.optModelUrls(params.modelUrl);
        }
        if (params.application != null) {
            criteriaBuilder.optApplication(params.application);
        }
        if (params.filters != null && !params.filters.isEmpty()) {
            criteriaBuilder.optFilters(params.filters);
        }

        return criteriaBuilder.build();
    }

    protected class Parameters {

        protected Class<?> inputType;
        protected Class<?> outputType;
        protected Application application;
        protected Map<String, String> filters;
        protected String modelUrl;

        /** constructs parameters object. */
        public Parameters() {}
        /**
         * gets the value of inputType.
         *
         * @return the inputType value.
         */
        public Class<?> getInputType() {
            if (inputType == null) {
                return Input.class;
            }
            return inputType;
        }
        /**
         * set the value of inputType.
         *
         * @param inputType the inputType to set
         */
        public void setInputType(Class<?> inputType) {
            this.inputType = inputType;
        }
        /**
         * gets the value of outputType.
         *
         * @return the outputType value.
         */
        public Class<?> getOutputType() {
            if (outputType == null) {
                return Output.class;
            }
            return outputType;
        }
        /**
         * set the value of outputType.
         *
         * @param outputType the outputType to set
         */
        public void setOutputType(Class<?> outputType) {
            this.outputType = outputType;
        }
        /**
         * gets the value of application.
         *
         * @return the application value.
         */
        public Application getApplication() {
            return application;
        }
        /**
         * set the value of application.
         *
         * @param application the application to set
         */
        public void setApplication(Application application) {
            this.application = application;
        }
        /**
         * gets the value of filters.
         *
         * @return the filters value.
         */
        public Map<String, String> getFilters() {
            return filters;
        }
        /**
         * set the value of filters.
         *
         * @param filters the filters to set
         */
        public void setFilters(Map<String, String> filters) {
            this.filters = filters;
        }
        /**
         * gets the value of modelUrl.
         *
         * @return the modelUrl value.
         */
        public String getModelUrl() {
            return modelUrl;
        }
        /**
         * set the value of modelUrl.
         *
         * @param modelUrl the modelUrl to set
         */
        public void setModelUrl(String modelUrl) {
            this.modelUrl = modelUrl;
        }
    }
}
