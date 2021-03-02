import React, { Component, useState, useEffect, useRef } from "react";
import Button from '@material-ui/core/Button';
import ReactDOM from 'react-dom';

import { makeStyles } from '@material-ui/core/styles';
import { theme } from '../css/useStyles'
import axios from 'axios'


const useFetch = (modelName) => {
	const [data, setData] = useState([]);

	useEffect(() => {
		async function fetchData() {
			axios.get("http://"+window.location.host+"/serving/models?modelName="+modelName)
				.then(function(response) {
					let appdata = Object.keys(response.data).map(function(key) {
						console.log(key)
						return {
							key: key,
							link: response.data[key]
						};
					});
					setData(appdata);
					console.log(appdata)
				})
				.catch(function(error) {
					console.log(error);
				})
				.then(function() {
					// always executed
				});

		}
		fetchData();
	}, ["http://"+window.location.host+"/serving/models?modelName="+modelName]);

	return data;
};



export default function ModelDownloadButtons(props) {
	const modelLinks = useFetch(props.modelName);
    return (
    		<>
    			{Object.keys(modelLinks).map((keys) => (
                   <Button href={modelLinks[keys].link}>Download {modelLinks[keys].key}</Button>

    				)
    			)}
    		</>
    	);

}