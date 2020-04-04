package ai.djl.ui.verticle;

import io.vertx.core.AbstractVerticle;
import io.vertx.core.http.HttpServer;
import io.vertx.core.logging.Logger;
import io.vertx.core.logging.LoggerFactory;
import io.vertx.ext.web.Router;
import io.vertx.ext.web.handler.BodyHandler;
import io.vertx.ext.web.handler.StaticHandler;
import io.vertx.ext.web.handler.sockjs.BridgeOptions;
import io.vertx.ext.bridge.PermittedOptions;
import io.vertx.ext.web.handler.sockjs.SockJSHandler;

import static ai.djl.ui.verticle.DataVerticle.*;

public class WebVerticle extends AbstractVerticle {

	private final Logger LOGGER = LoggerFactory.getLogger(WebVerticle.class.getCanonicalName());
	private final int port = 8080;

	@Override
	public void start() {
		HttpServer server = vertx.createHttpServer();
		Router router = Router.router(vertx);
		router.route().handler(BodyHandler.create());

		SockJSHandler sockJSHandler = SockJSHandler.create(vertx);
		BridgeOptions options = new BridgeOptions();

		options.addInboundPermitted(new PermittedOptions().setAddress(ADDRESS_TRAINER_REQUEST));

		options.addOutboundPermitted(new PermittedOptions().setAddress(ADDRESS_TRAINER));

		// Event bus
		router.mountSubRouter("/api/eventbus", sockJSHandler.bridge(options));

		// Static content (UI)
		router.route("/*").handler(StaticHandler.create());
		router.route("/*").handler(rc -> {
			if (!rc.currentRoute().getPath().startsWith("/api")) {
				rc.reroute("/index.html");
			}
		});

		server.requestHandler(router).listen(port, http -> {
			if (http.succeeded()) {
				LOGGER.info("HTTP server started on port {0}", port);
			} else {
				LOGGER.info("HTTP server failed on port {0}", port);
			}
		});
	}
}
