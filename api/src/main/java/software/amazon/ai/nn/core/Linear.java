package software.amazon.ai.nn.core;

import software.amazon.ai.Block;
import software.amazon.ai.engine.Engine;
import software.amazon.ai.ndarray.NDArray;

public interface Linear extends Block {

    NDArray forward(NDArray data);

    class Builder {

        private int units;

        private int inUnits;

        public int getUnits() {
            return units;
        }

        public Builder setUnits(int units) {
            this.units = units;
            return this;
        }

        public int getInUnits() {
            return inUnits;
        }

        public Builder setInUnits(int inUnits) {
            this.inUnits = inUnits;
            return this;
        }

        public Linear build() {
            return Engine.getInstance().getNNIndex().linear(units, inUnits);
        }
    }
}
