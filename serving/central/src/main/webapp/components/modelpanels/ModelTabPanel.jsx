import React, { Component, useState, useEffect, useRef } from "react";
import ReactDOM from 'react-dom';
import { makeStyles } from '@material-ui/core/styles';
import { theme } from '../../css/useStyles'

import Label from './Label/Label'
import Field from './Field/Field'
import Grid from '@material-ui/core/Grid';




export function ModelTabPanel(props) {
	const { model } = props;

console.log(model)

	return (
		<div>
			<Grid container spacing={3}>
				<Grid item xs={12}>
					<Label label={"Dataset"} value={model.dataset} />
				</Grid>
				<Field label={"Layers"} value={model.layers} />
				<Field label={"Backbone"} value={model.backbone} />
				<Grid item xs={12}>
				</Grid>
				
				<Field label={"Width"} value={model.width} />
				<Field label={"Height"} value={model.height} />
				<Field label={"Resize"} value={model.resize} />
				<Field label={"Rescale"} value={model.rescale} />
				
				{
					Object.keys(model.properties).map((key) => 
									 	<Field label={key} value={model.properties[key]} />
										
						)
				}
				
			</Grid>	
			
		</div>
	);
}

