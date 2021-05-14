import React, { Component, useState, useEffect, useRef } from "react";
import ReactDOM from 'react-dom';

import Label from '../Label/Label'

import Grid from '@material-ui/core/Grid';

function Field(props) {
	const { label, value } = props;
	
	
	return (
				<>
				{ typeof value !=='undefined' &&
					<Grid item xs={6}>
						<Label label={label} value={value} />
					</Grid>
				}
				</>
	);
}

export default Field;