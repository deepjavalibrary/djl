import React, { Component, useState, useEffect, useRef } from "react";
import ReactDOM from 'react-dom';

import SpeedDial from '@material-ui/lab/SpeedDial';
import SpeedDialIcon from '@material-ui/lab/SpeedDialIcon';
import SpeedDialAction from '@material-ui/lab/SpeedDialAction';
import ShareIcon from '@material-ui/icons/Share';

import { makeStyles } from '@material-ui/core/styles';
import { theme } from '../css/useStyles'
import axios from 'axios'


const useStyles = makeStyles((theme) => ({
	modelviewactions_root: {
	    position: 'absolute',
		right: '2em',
	   
  	},
	speedDial: {
	
	    position: 'relative',
	    '&.MuiSpeedDial-directionUp, &.MuiSpeedDial-directionLeft': {
	      bottom: theme.spacing(2),
	      right: theme.spacing(2),
	    },
	    '&.MuiSpeedDial-directionDown, &.MuiSpeedDial-directionRight': {
	      top: theme.spacing(2),
	      left: theme.spacing(2),
	    },
	},
}));

export default function ModelViewActions(props) {

	const classes = useStyles();

	const [open, setOpen] = React.useState(false);

	const handleClose = () => {
		setOpen(false);
	};

	const handleOpen = () => {
		setOpen(true);
	};
	
	const deploy = () => {
			axios.post("http://localhost:8080/serving/models?modelName="+props.modelName)
				.then(function(response) {
					console.log(response.data.message);
					alert(response.data.message)
				})
				.catch(function(error) {
					console.log(error);
				})
				.then(function() {
					// always executed
				});
			};

	return (
		<div className={classes.modelviewactions_root}>
			<SpeedDial
				ariaLabel="Model Actions"
				className={classes.speedDial}
				hidden={false}
				icon={<SpeedDialIcon />}
				onClose={handleClose}
				onOpen={handleOpen}
				open={open}
				direction={'down'}
			>

					<SpeedDialAction
						key={'Deploy'}
						icon={<ShareIcon />}
						tooltipTitle={'deploy'}
						onClick={deploy}
				
					/>
			</SpeedDial>
		</div>
	);
}