import React, { Component, useState, useEffect, useRef } from "react";
import ReactDOM from 'react-dom';

import Grid from '@material-ui/core/Grid';
import Tabs from '@material-ui/core/Tabs';
import Tab from '@material-ui/core/Tab';
import Box from '@material-ui/core/Box';

import { makeStyles } from '@material-ui/core/styles';
import { theme } from '../../css/useStyles'

const useStyles = makeStyles((theme) => ({
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
	root: {
	    flexGrow: 1,
	    backgroundColor: theme.palette.background.paper,
	    display: 'flex',
	    height: 224,
  	},

}));



function a11yProps(index) {
	return {
		id: `vertical-tab-${index}`,
		'aria-controls': `vertical-tabpanel-${index}`,
	};
}

function TabPanel(props) {
	const { children, tabs, ...other } = props;
	const classes = useStyles();

	const [currentIndex, setCurrentIndex] = React.useState(0);

	const handleTabChange = (event, newIndex) => {
		setCurrentIndex(newIndex);
	};

	let tabIndex=0;

	return (
		<div className={classes.root}>
					<Tabs
						orientation="vertical"
						variant="scrollable"
						value={currentIndex}
						onChange={handleTabChange}
				
						aria-label="{props.model.name}">
						{
							
							tabs.map( title=><Tab label={title} {...a11yProps(tabIndex++)} /> ) 
						}

					</Tabs>
	
						{children.filter((child, index) => index == currentIndex)
							.map(child => <div role="tabpanel"><Box p={3}>{child}</Box></div>)
						}
		</div>


	);
}

export default TabPanel;