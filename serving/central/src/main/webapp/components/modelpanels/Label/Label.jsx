import React, { Component, useState, useEffect, useRef } from "react";
import ReactDOM from 'react-dom';

import TextField from '@material-ui/core/TextField';

function Label(props) {
	const { label, value } = props;
	
	
	return (
			<TextField
				id={label}
				label={label}
				key={value}
				fullWidth={true}
				size={"small"}
				defaultValue={value}
				InputProps={{
							readOnly: true,
							}}					
			/>
	);
}

export default Label;