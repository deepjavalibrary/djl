import React, { Component, useState, useEffect, useRef } from "react";
import Button from '@material-ui/core/Button';
import ReactDOM from 'react-dom';

import { makeStyles } from '@material-ui/core/styles';
import axios from 'axios'


const useFetch = (model) => {
	const [data, setData] = useState([]);

	useEffect(() => {
		async function fetchData() {

			axios.get("http://"+window.location.host+"/serving/models?modelName="+model.name+"&artifactId="+model.metadata.artifactId+"&groupId="+model.metadata.groupId)
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
	}, [model.modelName,model.metadata.artifactId,model.metadata.groupId]);

	return data;
};



export default function ModelDownloadButtons(props) {
	const modelUris = useFetch(props.model);
    return (
    		<>
    			{Object.keys(modelUris).map((keys) => (
                   <Button href={modelUris[keys].link}>Download {modelUris[keys].key}</Button>

    				)
    			)}
    		</>
    );
}
