import React, { Component, useState, useEffect, useRef } from "react";
import ReactDOM from 'react-dom';
import TextField from '@material-ui/core/TextField';
import { makeStyles } from '@material-ui/core/styles';

const useStyles = makeStyles((theme) => ({
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

}));


export default function DynForm(props) {

	const classes = useStyles();
	
	
	
	return (
		<>
			{Object.keys(props.data).filter(key=>key!=='licenses').map((key) => (
					<div >
						<TextField
							id={key}
							label={key}
							key={props.data[key]}
							fullWidth={true}
							defaultValue={props.data[key]}
							InputProps={{
								readOnly: true,
							}}

							className={classes.textfield}
						/>
					</div>
				)
			)}
		</>
	);
}