package ai.djl.examples.inference.owlv2; 

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.Pair;

/**
 * TestOwlv2ImageTextTranslator
 */
public class TestOwlv2ImageTextTranslator {

	@Test
	public void testPredict() throws Exception {
		Image image = ImageFactory.getInstance().fromFile(Paths.get("./src/test/resources/cats.jpg"));
		List<String> labels = Arrays.asList("cat's ear", "cat's tail");
		Pair<Image, List<String>> input = new Pair<>(image, labels);
		Owlv2ObjectDetector detector = new Owlv2ObjectDetector();
		detector.predict(input);
	}

	@Test
	public void testStackArrays_andResultingArrayHasExpectedValues() {
		NDManager manager = NDManager.newBaseManager();
		NDArray a = manager.zeros(new Shape(7));
		NDArray b = manager.ones(new Shape(7));
		NDArray stacked = Owlv2ImageTextTranslator.stackArrays(Arrays.asList(a, b));
		assertEquals(stacked.getShape(), new Shape(2, 7));
	}

	@Test
	public void testPadMaxLength_andAllArraysAreLengthOfLongest() {
		NDManager manager = NDManager.newBaseManager();
		NDArray a = manager.zeros(new Shape(5));
		NDArray b = manager.zeros(new Shape(7));
		NDArray c = manager.zeros(new Shape(10));
		List<NDArray> arrays = Owlv2ImageTextTranslator.padMaxLength(manager, Arrays.asList(a, b, c));
		assertTrue(
				arrays.stream().mapToInt(arr -> (int) arr.size()).allMatch(i -> i == 10));
	}

	
}
