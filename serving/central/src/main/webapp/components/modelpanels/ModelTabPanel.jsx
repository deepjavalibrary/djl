import React, { Component, useState, useEffect, useRef } from "react";
import ReactDOM from 'react-dom';
import { makeStyles } from '@material-ui/core/styles';
import { theme } from '../../css/useStyles'

import Label from './Label/Label'

import Grid from '@material-ui/core/Grid';




export function ModelTabPanel(props) {
	const { model } = props;


	return (
		<div>
			<Grid container spacing={3}>
				<Grid item xs={12}>
					<Label label={"Dataset"} value={model.dataset} />
				</Grid>
				<Grid item xs={6}>
					<Label label={"Layers"} value={model.layers} />
				</Grid>
				<Grid item xs={6}>
					<Label label={"Backbone"} value={model.backbone} />
				</Grid>
				<Grid item xs={12}>
				</Grid>
				
				<Grid item xs={6}>
					<Label label={"Width"} value={model.width} />
				</Grid>
				<Grid item xs={6}>
					<Label label={"Height"} value={model.height} />
				</Grid>
				<Grid item xs={6}>
					<Label label={"Resize"} value={model.resize} />
				</Grid>
				<Grid item xs={6}>
					<Label label={"Rescale"} value={model.rescale} />
				</Grid>
				<Grid item xs={6}>
					<Label label={"Threshold"} value={model.threshold} />
				</Grid>
				<Grid item xs={6}>
				</Grid>
				<Grid item xs={6}>
					<Label label={"Synset FileName"} value={model.synsetFileName} />
				</Grid>
				
				
			</Grid>	
			
		</div>
	);
}

