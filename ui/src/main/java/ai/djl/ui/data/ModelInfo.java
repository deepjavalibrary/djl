package ai.djl.ui.data;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ModelInfo {
	
	private String name;
	private String block;

}
