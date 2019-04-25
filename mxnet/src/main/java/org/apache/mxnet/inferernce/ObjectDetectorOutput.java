/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package org.apache.mxnet.inferernce;

public class ObjectDetectorOutput {

    private String className;
    private float[] args;

    public ObjectDetectorOutput(String className, float[] args) {
        this.className = className;
        this.args = args;
    }

    public String getClassName() {
        return className;
    }

    public float getProbability() {
        return args[0];
    }

    public float getXMin() {
        return args[1];
    }

    public float getXMax() {
        return args[2];
    }

    public float getYMin() {
        return args[3];
    }

    public float getYMax() {
        return args[4];
    }
}
