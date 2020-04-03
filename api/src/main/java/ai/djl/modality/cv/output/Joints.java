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
package ai.djl.modality.cv.output;

import java.io.Serializable;
import java.util.List;

/**
 * A result of all joints found during Human Pose Estimation on a single image.
 *
 * @see <a href="https://en.wikipedia.org/wiki/Articulated_body_pose_estimation">Wikipedia</a>
 */
public class Joints implements Serializable {

    private static final long serialVersionUID = 1L;
    private List<Joint> joints;

    /**
     * Constructs the {@code Joints} with the provided joints.
     *
     * @param joints the joints
     */
    public Joints(List<Joint> joints) {
        this.joints = joints;
    }

    /**
     * Gets the joints for the image.
     *
     * @return the list of joints
     */
    public List<Joint> getJoints() {
        return joints;
    }

    /** {@inheritDoc} */
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

    /**
     * A joint that was detected using Human Pose Estimation on an image.
     *
     * @see Joints
     */
    public static class Joint extends Point {
        private static final long serialVersionUID = 1L;
        private double confidence;

        /**
         * Constructs a Joint with given data.
         *
         * @param x the x coordinate of the joint
         * @param y the y coordinate of the joint
         * @param confidence the confidence probability for the joint
         */
        public Joint(double x, double y, double confidence) {
            super(x, y);
            this.confidence = confidence;
        }

        /**
         * Returns the confidence probability for the joint.
         *
         * @return the confidence
         */
        public double getConfidence() {
            return confidence;
        }

        /** {@inheritDoc} */
        @Override
        public String toString() {
            return String.format(
                    "Joint [x=%.3f, y=%.3f], confidence: %.4f", getX(), getY(), getConfidence());
        }
    }
}
