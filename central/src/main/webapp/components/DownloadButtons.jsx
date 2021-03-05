import React, { Component, useState, useEffect, useRef } from "react";
import Button from '@material-ui/core/Button';
import ReactDOM from 'react-dom';

import { makeStyles } from '@material-ui/core/styles';
import axios from 'axios'


const useFetch = (modelName) => {
	const [data, setData] = useState([]);

	useEffect(() => {
		async function fetchData() {
			axios.get("http://"+window.location.host+"/serving/models?modelName="+modelName)
				.then(function(response) {
					let appdata = Object.keys(response.data).map(function(key) {
						return {
							key: key,
							link: response.data[key]
						};
					});
					setData(appdata);
					console.log(appdata)
				})
		}
		fetchData();
	}, [modelName]);

	return data;
};



export default function ModelDownloadButtons(props) {
	const modelUris = useFetch(props.modelName);
    return (
    		<>
    			{Object.keys(modelUris).map((keys) => (
                   <Button href={modelUris[keys].link}>Download {modelUris[keys].key}</Button>

    				)
    			)}
    		</>
    );
}
