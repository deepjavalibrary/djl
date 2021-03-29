
var path = require('path');

module.exports = {
	// context: path.resolve(__dirname, ''),
	entry: {
		main: './src/main/webapp/Main.jsx'
	},
	devtool: 'source-map',
	module: {
		rules: [{
			test: /\.(js|jsx)$/,
			exclude: /node_modules/,
			loader: "babel-loader",
			options: {
				presets: ['@babel/preset-env', '@babel/preset-react']
			}
		}, {
			test: /\.css$/,
			exclude: /node_modules/,
			use: ['style-loader', 'css-loader'],

		},
		{
			test: /\.(png|svg|jpg|jpeg|gif)$/i,
			exclude: /node_modules/,
			type: 'asset/resource',

		},
		{
			test: /\.(woff|woff2|eot|ttf|otf)$/i,
			type: 'asset/resource',
		}
	]},
	resolve: {
		extensions: ['.js', '.jsx', '.css']
	},
	devServer: {
		contentBase: path.join(__dirname, 'src/main/resources/main/static/'),
		hot: true,
		liveReload: true,
		compress: true,
		port: 9000
	}
};
