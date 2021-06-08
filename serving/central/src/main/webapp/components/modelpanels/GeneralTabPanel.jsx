import React, { Component, useState, useEffect, useRef } from "react";
import ReactDOM from 'react-dom';
import { makeStyles } from '@material-ui/core/styles';
import { theme } from '../../css/useStyles'

import Label from './Label/Label'

import Grid from '@material-ui/core/Grid';


/**
	private String groupId;
	private String artifactId;
	private String version;
	private boolean snapshot;
	private String decription;
	private String website;
	private URI repositoryURI;
	private Date lastUpdated;
	private String resourceType;
	private Map<String,String> licenses; */

export function GeneralTabPanel(props) {
	const { model } = props;


	return (
		<div>
			<Grid container spacing={3}>
				<Grid item xs={6}>
					<Label label={"Name"} value={model.name} />
				</Grid>
				<Grid item xs={6}>
					<Label label={"Version"} value={model.version} />
				</Grid>
				<Grid item xs={12}>
					<Label label={"Description"} value={model.description} />
				</Grid>
				<Grid item xs={12}>
					<Label label={"Website"} value={model.website} />
				</Grid>
				<Grid item xs={12}>
					<Label label={"Repository URI"} value={model.repositoryURI} />
				</Grid>
				<Grid item xs={12}></Grid>
				{ typeof model.licenses==='undefined' ? <div/> : Object.keys(model.licenses).map( k=> <> <Grid item xs={6}><Label label={"License"} value={k} /> </Grid> <Grid item xs={6}><Label label={"License URL"} value={model.licenses[k]} /> </Grid> </> ) }


			</Grid>	
			
		</div>
	);
}

