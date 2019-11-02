/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.cv;

import java.util.List;

public class Joints {

    private List<Joint> joints;

    public Joints(List<Joint> joints) {
        this.joints = joints;
    }

    public List<Joint> getJoints() {
        return joints;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(4000);
        sb.append("\n[\t");
        boolean first = true;
        for (Joint joint : joints) {
            if (first) {
                first = false;
            } else {
                sb.append(",\n\t");
            }
            sb.append(joint);
        }
        sb.append("\n]");
        return sb.toString();
    }

    public static class Joint extends Point {
        private double confidence;

        public Joint(double x, double y, double confidence) {
            super(x, y);
            this.confidence = confidence;
        }

        public double getConfidence() {
            return confidence;
        }

        @Override
        public String toString() {
            return String.format(
                    "Joint [x=%.3f, y=%.3f], confidence: %.4f", getX(), getY(), getConfidence());
        }
    }
}
