package ai.djl.ui.data;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class MetricInfo {

	private String name;
	private Integer x;
	private Float y;

}
