package software.amazon.ai;

import java.util.Collections;
import java.util.List;
import software.amazon.ai.ndarray.NDArray;

/**
 * An interface representing an initialization method.
 *
 * <p>Used to initialize the {@link NDArray} parameters stored within a {@link Block}.
 */
public interface Initializer {

    /**
     * Initializes a single NDArray
     *
     * @param array the NDArray to initialize
     */
    default void initialize(NDArray array) {
        initialize(Collections.singletonList(array));
    }

    /**
     * Initializes a list of NDArrays
     *
     * @param parameters the NDArrays to initialize
     */
    void initialize(List<NDArray> parameters);
}
