import React, { Component, useState, useEffect, useRef } from "react";
import ReactDOM from 'react-dom';
// import useStyles from '../css/useStyles';

import Paper from '@material-ui/core/Paper';
import GridList from '@material-ui/core/GridList';
import GridListTile from '@material-ui/core/GridListTile';
import GridListTileBar from '@material-ui/core/GridListTileBar';
import ListSubheader from '@material-ui/core/ListSubheader';
import IconButton from '@material-ui/core/IconButton';
import InfoIcon from '@material-ui/icons/Info';

import TextField from '@material-ui/core/TextField';
import Tabs from '@material-ui/core/Tabs';
import Tab from '@material-ui/core/Tab';
import Typography from '@material-ui/core/Typography';
import Box from '@material-ui/core/Box';
import Chip from '@material-ui/core/Chip';
import Divider from '@material-ui/core/Divider';

import DynForm from './DynForm';
import ModelViewActions from './ModelViewActions';

import axios from 'axios'



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


	textfield: {
		"& .MuiFormLabel-root": {
			color: "rgb(250, 104, 33)",
			padding: 0,
			fontSize: "1rem",
			fontFamily: "Roboto, Helvetica, Arial, sans-serif",
			fontWeight: 400,
			lineHeight: 1,
			letterSpacing: "0.00938em",
		},
		"& .MuiInputBase-input": {
			color: "rgb(14, 14, 12)",
		},

		"& .MuiInput-underline:before": {
			left: 0,
			right: 0,
			bottom: 0,
			content: "\\00a0",
			position: "absolute",
			transition: "border-bottom-color 200ms cubic-bezier(0.4, 0, 0.2, 1) 0ms",
			borderBottom: "1px solid rgb(228, 168, 64 , 1)",
			pointerEvents: "none",
		}

	},
	tabs: {
		borderRight: `1px solid ${theme.dividerColor}`,
	},
	tabbar: {
		flexGrow: 1,

		display: 'flex',
	},
	tabpanel: {
		flex: '2 1 auto',
		alignSelf: 'stretch',
	}, 
	dynform: {
		width: '100%',
		marginLeft: "2em",
	},
}));

function TabPanel(props) {
	const { children, value, index,  ...other } = props;
	
	
	return (
		<div
			role="tabpanel"
			hidden={value !== index}
			id={`vertical-tabpanel-${index}`}
			aria-labelledby={`vertical-tab-${index}`}
			{...other}
		>
			{value === index && (
				<Box p={3} >
					<Typography>{children}</Typography>
				</Box>
			)}
		</div>
	);
}


function a11yProps(index) {
	return {
		id: `vertical-tab-${index}`,
		'aria-controls': `vertical-tabpanel-${index}`,
	};
}


export default function ModelView(props) {

	const classes = useStyles();
	const myRef = useRef(null);

	const [index, setIndex] = React.useState(0);

	const handleTabChange = (event, newIndex) => {
		setIndex(newIndex);
	};

	const executeScroll = () => myRef.current.scrollIntoView()

	useEffect(() => {
		executeScroll();
	}, [props.model])

	console.log(props.model);

	let noArguments = {
		width: 0,
		height: 0,
		resize: null,
	};

	let noParameters = {
		uri: "N/A",
		sha1Hash: "N/A",
		size: 0
	};

	let noSynset = {
		name: "N/A",
		Hash: "N/A",
		size: 0,
		uri: "N/A",
	};

	let data;
	return (

		<div className={classes.model_view_root}>
			
			<Paper ref={myRef} elevation={3} className={classes.model_view_paper} >
				<ModelViewActions modelName={props.model.name} modelUri={props.model.files.parameters.uri}/>
				<h2>{props.model.name}</h2>
				<h3>{props.model.metadata.groupId}:{props.model.metadata.artifactId}:{props.model.version}</h3>
				<Chip size="small" label={props.model.properties.dataset} />
				<Chip size="small" label={props.model.version} />
				<div className={classes.tabbar}>
					<Tabs
						orientation="vertical"
						variant="scrollable"
						value={index}
						onChange={handleTabChange}
						className={classes.tabs}
						aria-label="{props.model.name}">

						<Tab label="General" {...a11yProps(0)} />
						<Tab label="Metadata" {...a11yProps(1)} />
						<Tab label="Properties" {...a11yProps(2)} />
						<Tab label="Arguments" {...a11yProps(3)} />
						<Tab label="Parameters" {...a11yProps(4)} />
						<Tab label="Synset" {...a11yProps(5)} />
					</Tabs>

					<TabPanel value={index} index={0} className={classes.tabpanel}>
						<>
							{Object.keys(props.model).filter((key) => !(typeof props.model[key] === 'object')).map((key) => (
								<div >
									<TextField
										id={key}
										label={key}
										key={props.model[key]}
										fullWidth={true}
										defaultValue={props.model[key]}
										InputProps={{
											readOnly: true,
										}}

										className={classes.textfield}
									/>
								</div >
							)
							)}
							<div >
								<TextField
									id={'dataset1'}
									label={'dataset'}
									key={props.model.properties.dataset}
									fullWidth={true}
									defaultValue={props.model.properties.dataset}
									InputProps={{
										readOnly: true,
									}}

									className={classes.textfield}
								/>
							</div >

						</>
					</TabPanel>
					<TabPanel value={index} index={1} className={classes.tabpanel}>
						<DynForm data={props.model.metadata} />
						<h4>Licenses</h4>
						<DynForm data={props.model.metadata.licenses}/>
					</TabPanel>
					<TabPanel value={index} index={2} className={classes.tabpanel}>
						<DynForm data={props.model.properties} />
					</TabPanel>
					<TabPanel value={index} index={3} className={classes.tabpanel}>
						{props.model.arguments
							? <DynForm data={props.model.arguments} />
							: <DynForm data={noArguments}/>
						}
					</TabPanel>
					<TabPanel value={index} index={4} className={classes.tabpanel}>
						{props.model.files.parameters
							? <DynForm data={props.model.files.parameters}/>
							: <DynForm data={noParameters}/>
						}
					</TabPanel>
					<TabPanel value={index} index={5} className={classes.tabpanel}>
						{props.model.files.synset
							? <DynForm data={props.model.files.synset}/>
							: <DynForm data={noSynset}/>
						}
					</TabPanel>
					
				</div>
				
			</Paper>
		</div>

	);
}
