import React, { Component, useState, useEffect, useRef } from "react";
import ReactDOM from 'react-dom';
import { makeStyles } from '@material-ui/core/styles';
import { theme } from '../../css/useStyles'

import Label from './Label/Label'
import Button from '@material-ui/core/Button';
import Grid from '@material-ui/core/Grid';




export function FilesTabPanel(props) {
	const { model } = props;


	return (
		<div>
			<Grid container spacing={3}>
			
				{
					model.files.map( file=> 
						<>
							<Grid item xs={10}>
								<h3>{file.key}</h3>
							</Grid>
							<Grid item xs={2}>
								<Button href={file.downloadLink}>Download</Button>
							</Grid>
							<Grid item xs={6}>
								<Label label={"Filename"} value={file.name} />
							</Grid>
							<Grid item xs={6}>
								<Label label={"URI"} value={file.uri} />
							</Grid>
							<Grid item xs={6}>
								<Label label={"sha1Hash"} value={file.sha1Hash} />
							</Grid>
							<Grid item xs={6}>
								<Label label={"Size"} value={file.size} />
							</Grid>
						</>

					)
				}

				
			</Grid>	
			
		</div>
	);
}

