package ai.djl.ui;

import ai.djl.ui.verticle.DataVerticle;
import ai.djl.ui.verticle.WebVerticle;
import io.vertx.core.AbstractVerticle;
import io.vertx.core.logging.Logger;
import io.vertx.core.logging.LoggerFactory;

public class Main extends AbstractVerticle {
	private final Logger LOGGER = LoggerFactory.getLogger(Main.class );

	@Override
	public void start() {
		deployVerticle(DataVerticle.class.getName());
		deployVerticle(WebVerticle.class.getName());
	}
	
	protected void deployVerticle(String className) {
	  vertx.deployVerticle(className, res -> {
	    if (res.succeeded()) {
			LOGGER.info("Deployed verticle: {}", className);
	    } else {
			LOGGER.info("Error deploying verticle: {}", className, res.cause());
	    }
	  });
	}

}
