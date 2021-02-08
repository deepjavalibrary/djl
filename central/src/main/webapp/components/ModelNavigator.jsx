import React, { useState, useEffect } from "react";
import { makeStyles } from '@material-ui/core/styles';
import TreeView from '@material-ui/lab/TreeView';
import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import ChevronRightIcon from '@material-ui/icons/ChevronRight';
import TreeItem from '@material-ui/lab/TreeItem';
import ModelView from './ModelView';

import axios from 'axios'
import DynForm from "./DynForm";

const useStyles = makeStyles({
	view_root: {
		display: "flex",
		flexDirection: "row",
		flexWrap: "nowrap",
		justifyContent: "flex-start",
		alignContent: "stretch",
		alignItems: "flex-start",
		margin: '3em',
	},
	navigator_root: {
		order: 0,
		flex: "0 1 auto",
		alignSelf: "stretch",
		maxHeight: '600px',
		overflowY: "auto",

	},

	model_root: {
		order: 0,
		flex: "2 1 auto",
		alignSelf: "stretch",
	//			'-webkit-box-shadow': '0px 10px 13px -7px #000000, 5px 5px 15px 5px rgba(11,60,93,0)',
	//	boxShadow: '0px 10px 13px -7px #000000, 5px 5px 15px 5px rgba(11,60,93,0)',
	},
});




const useFetch = (url) => {
	const [data, setData] = useState([]);

	// empty array as second argument equivalent to componentDidMount
	useEffect(() => {
		async function fetchData() {
			axios.get(url)
				.then(function(response) {
					console.log(response)

					let appdata = Object.keys(response.data).map(function(key) {
						console.log(key)
						return {
							key: key,
							title: key,
							models: response.data[key]
						};
					});
					console.log(appdata)
					setData(appdata);
				})
				.catch(function(error) {
					console.log(error);
				})
				.then(function() {
					// always executed
				});

		}
		fetchData();
	}, [url]);

	return data;
};





export default function ModelNavigator(props) {

	const classes = useStyles();

	const URL = 'http://localhost:8080/models';
	const modelZooData = useFetch(URL);

	const [model, setModel] = useState(null);

	return (
		<>

			<div className={classes.view_root}>
				<div className={classes.navigator_root}>
				<TreeView
					
					defaultCollapseIcon={<ExpandMoreIcon />}
					defaultExpandIcon={<ChevronRightIcon />}
				>
					{modelZooData.map((application) => (
						<TreeItem nodeId={application.key} label={application.title}>
							{application.models.map((model) => (
								<TreeItem nodeId={model.name} label={model.name} onLabelClick={() => setModel(model)}>
								</TreeItem>
							))}
						</TreeItem>
					))}

				</TreeView>
				</div>

				{model != null &&
					<div className={classes.model_root}>
						<ModelView model={model} />
					</div>
				}
			</div>
		</>
	);
}