import React, { Component } from "react";
import ReactDOM from 'react-dom';

import useStyles from '../css/useStyles';

export default function PageTitle(props) {

	const classes = useStyles();

	return (
		<div className={classes.pageTitle}>{props.title}</div>
	);
}