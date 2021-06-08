

import regeneratorRuntime from "regenerator-runtime";

import React, { Component } from "react";
import ReactDOM from 'react-dom';

import CssBaseline from '@material-ui/core/CssBaseline';
import Typography from '@material-ui/core/Typography';
import Container from '@material-ui/core/Container';

import PageTitle from './components/PageTitle';
import ModelNavigator from './components/ModelNavigator';

import useStyles from './css/useStyles';

import './css/style.css'



export default function Main() {

	const classes = useStyles();

	return (
		<div className={classes.appRoot}>


			<div className={classes.content}>
				<CssBaseline />
				<Typography>
					<Container>

								<ModelNavigator />
					</Container>
				</Typography>
			</div>
			<PageTitle title="MODEL|zoo"></PageTitle>
		</div>
	);

}

ReactDOM.render(
	<Main />,
	document.getElementById('react-mountpoint')
);
