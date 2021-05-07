import React, { Component, useState, useEffect, useRef } from "react";
import ReactDOM from 'react-dom';


import GridList from '@material-ui/core/GridList';
import GridListTile from '@material-ui/core/GridListTile';
import GridListTileBar from '@material-ui/core/GridListTileBar';
import ListSubheader from '@material-ui/core/ListSubheader';
import IconButton from '@material-ui/core/IconButton';
import InfoIcon from '@material-ui/icons/Info';
import Box from '@material-ui/core/Box';
import Typography from '@material-ui/core/Typography';
import TextField from '@material-ui/core/TextField';



import Chip from '@material-ui/core/Chip';
import Divider from '@material-ui/core/Divider';
import Grid from '@material-ui/core/Grid';


import { GeneralTabPanel } from './modelpanels/GeneralTabPanel'
import { ModelTabPanel } from './modelpanels/ModelTabPanel'
import { FilesTabPanel } from './modelpanels/FilesTabPanel'


import { TabPanel } from './TabPanel/'



import axios from 'axios'

import { fetchData } from './network.functions'


import { makeStyles } from '@material-ui/core/styles';
import { theme } from '../css/useStyles'


const useStyles = makeStyles((theme) => ({
	model_view_root : {
		padding: "2em",
		marginTop: "2em",
		backgroundColor: "white",
		color: "black",
		minHeight: "100%",
    	overflow: "scroll",
    	borderRadius: "25",
    	'-webkit-box-shadow': '0px 10px 13px -7px #000000, 5px 5px 15px 5px rgba(11,60,93,0)',
		boxShadow: '0px 10px 13px -7px #000000, 5px 5px 15px 5px rgba(11,60,93,0)',
	},
	header: {
	//	transform: "rotate(-10deg)",
		background: theme.overlayColor,
		borderRadius: "5",
	},
}));







export default function ModelView(props) {

	const URL = 'http://' + window.location.hostname + ':' + window.location.port + '/modelzoo/models/' + props.modelRef.groupId + ":" + props.modelRef.artifactId + ":" + props.modelRef.version + ":" + props.modelRef.name;
	const model = fetchData(URL);

	const classes = useStyles(theme);
	const myRef = useRef(null);



	return (

		<>

			{model != null &&
				<div className={classes.model_view_root}>
							<Typography>
								<Box className={classes.header}>
								<h2>{model.name} - {model.version}</h2>
								<h3>{model.groupId}:{model.artifactId}:{model.version}</h3>
								<Chip size="small" label={model.version} />
								<Chip size="small" label={model.resourceType} />
								</Box>
							</Typography>
							
					<Grid container spacing={2} >
						<Grid item xs={10}>
							<TabPanel tabs={["General", "Model", "Files"]} >
								<GeneralTabPanel model={model} />
								<ModelTabPanel model={model} />
								<FilesTabPanel model={model} />
							</TabPanel>
						</Grid>
						<Grid item xs={2}>
						</Grid>
					
					</Grid>
			
				</div>
			
			}

		</>

	);
}
