import React, { Component, useState, useEffect, useRef } from "react";
import ReactDOM from 'react-dom';


import Paper from '@material-ui/core/Paper';
import GridList from '@material-ui/core/GridList';
import GridListTile from '@material-ui/core/GridListTile';
import GridListTileBar from '@material-ui/core/GridListTileBar';
import ListSubheader from '@material-ui/core/ListSubheader';
import IconButton from '@material-ui/core/IconButton';
import InfoIcon from '@material-ui/icons/Info';

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
	model_view_root: {

		flexWrap: 'wrap',
		justifyContent: 'space-around',
		overflow: 'hidden',

		order: 3,
		flex: '2 1 auto',
		alignSelf: 'stretch',

	},
	model_view_paper: {
		minHeight: '600px',
		padding: '20px',
		overflowY: "auto",
		marginRight: 'auto',
		marginLeft: '2em',
		marginBottom: '5vh',
	},



}));







export default function ModelView(props) {

	const URL = 'http://' + window.location.hostname + ':' + window.location.port + '/modelzoo/models/' + props.modelRef.groupId + ":" + props.modelRef.artifactId + ":" + props.modelRef.version + ":" + props.modelRef.name;
	const model = fetchData(URL);

	const classes = useStyles();
	const myRef = useRef(null);



	return (

		<div className={classes.model_view_root}>

			{model != null &&
				<Paper ref={myRef} elevation={3} className={classes.model_view_paper} >
					<Grid container spacing={3}>
						<Grid item xs={12}>
							<h2>{model.name} - {model.version}</h2>
							<h3>{model.groupId}:{model.artifactId}:{model.version}</h3>
							<Chip size="small" label={model.version} />
							<Chip size="small" label={model.resourceType} />
						</Grid>
						<Grid item xs={8}>
							<TabPanel tabs={["General", "Model", "Files"]} >
								<GeneralTabPanel model={model} />
								<ModelTabPanel model={model} />
								<FilesTabPanel model={model} />
							</TabPanel>
						</Grid>
						<Grid item xs={4}>
						</Grid>
					</Grid>


				</Paper>
			}

		</div>

	);
}
