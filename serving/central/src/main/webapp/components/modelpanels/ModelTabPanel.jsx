import React, { Component, useState, useEffect, useRef } from "react";
import ReactDOM from 'react-dom';
import { makeStyles } from '@material-ui/core/styles';
import { theme } from '../../css/useStyles'

import Label from './Label/Label'

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
				<Grid item xs={6}>
					<Label label={"Layers"} value={model.layers} />
				</Grid>
				<Grid item xs={6}>
					<Label label={"Backbone"} value={model.backbone} />
				</Grid>
				<Grid item xs={12}>
				</Grid>
				
				{ typeof model.width !=='undefined' &&
					<Grid item xs={6}>
						<Label label={"Width"} value={model.width} />
					</Grid>
				}
				{ typeof model.height !=='undefined' &&
					<Grid item xs={6}>
						<Label label={"Height"} value={model.height} />
					</Grid>
				}
				{ typeof model.resize !=='undefined' &&
					<Grid item xs={6}>
						<Label label={"Resize"} value={model.resize} />
					</Grid>
				}
				{ typeof model.rescale !=='undefined' &&
					<Grid item xs={6}>
						<Label label={"Rescale"} value={model.rescale} />
					</Grid>
				}
				
				{
					Object.keys(model.properties).map((key) => {
									return 	<Grid item xs={6}>
												<Label label={key} value={model.properties[key]} />
											</Grid>
								}		
						)
				}
				
			</Grid>	
			
		</div>
	);
}

