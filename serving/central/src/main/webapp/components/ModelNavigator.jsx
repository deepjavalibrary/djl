import React, { useState, useEffect } from "react";
import { makeStyles } from '@material-ui/core/styles';
import Input from '@material-ui/core/Input';
import TextField from '@material-ui/core/TextField';
import TreeView from '@material-ui/lab/TreeView';
import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import ChevronRightIcon from '@material-ui/icons/ChevronRight';
import TreeItem from '@material-ui/lab/TreeItem';
import ModelView from './ModelView';

import axios from 'axios'

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
	inputComponent: {
        height: '15%',
        width:  '33%',
        backgroundColor: 'white',
        boxShadow: '0 1px 0 0 rgba(170,170,170,0.01)'
    },
    inputText: {
        color: '#D3D4D0',
        fontSize: '16px',
        letterSpacing: '0.5px',
        lineHeight: '28px',
        textAlign: 'center',
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
	const [modelList, setModelList] = useState([]);
	const [nameValue, setNameValue] = useState('');
	const [applicationValue, setApplicationValue] = useState('');
	const [versionValue, setVersionValue] = useState('');
	const [versionList, setVersionList] = useState([]);

    const filteredModels =
        modelZooData.map((application) => (
            application.models.filter((model) => {
                if ((versionValue == '') || (versionValue == model.version)){
                    if ((applicationValue == '') || (applicationValue == application.key)){
                        return model.name.toLowerCase().includes(nameValue.toLowerCase());
                    }
                }
            })
        ))

	const modelFilterOnChange = (event) => {
        setNameValue(event.target.value);
        setModelList(filteredModels)

    };

    const modelApplicationFilterOnChange = (event) => {
        setApplicationValue(event.target.value);
    };

    const modelVersionFilterOnChange = (event) => {
        setVersionValue(event.target.value);
    };

    function handleAdd(version) {
        if (versionList.includes(version) == false){
            const newList = versionList;
            setVersionList(newList.concat(version));
        }
      }

	return (
		<>
           <div className={classes.view_root}>
                <div className={classes.navigator_root}>
                    <TreeView
                        defaultExpanded={['Model Searching']}
                        defaultCollapseIcon={<ExpandMoreIcon />}
                        defaultExpandIcon={<ChevronRightIcon />}
                    >
                        <TreeItem nodeId="Model Searching" label="Model Search">
                            <TreeView
                                defaultCollapseIcon={<ExpandMoreIcon />}
                                defaultExpandIcon={<ChevronRightIcon />}
                            >
                                <TreeItem nodeId="Name" label="Name">
                                    <TextField
                                        id="name-search"
                                        label="Enter Name"
                                        value={nameValue}
                                        onChange={modelFilterOnChange}
                                        inputProps={{style: { backgroundColor:"white"},}}
                                    />
                                </TreeItem>
                                <TreeItem nodeId="Version" label="Version">
                                    <div onChange={modelVersionFilterOnChange}>
                                        {modelZooData.map((application) => (
                                            application.models.map((model) => (
                                                    handleAdd(model.version)
                                            ))
                                        ))}

                                        {versionList.map((version) => (
                                            <button disabled={versionValue==version} value={version} onClick={modelVersionFilterOnChange} > {version} </button>
                                        ))}
                                        <button value="" onClick={modelVersionFilterOnChange} > Clear </button>
                                    </div>
                                </TreeItem>
                                <TreeItem nodeId="Application" label="Application">
                                    <div onChange={modelApplicationFilterOnChange}>
                                        {modelZooData.map((application) => (
                                            <button
                                                disabled={applicationValue == application.title}
                                                value={application.title}
                                                onClick={modelApplicationFilterOnChange}
                                            >

                                                {application.title}

                                            </button>
                                        ))}
                                       <button value='' onClick={modelApplicationFilterOnChange} > Clear </button>
                                    </div>
                                </TreeItem>
                            </TreeView>
                            <button onClick={modelFilterOnChange} > Search </button>
                        </TreeItem>
                    </TreeView>

				<TreeView

					defaultCollapseIcon={<ExpandMoreIcon />}
					defaultExpandIcon={<ChevronRightIcon />}
				>
				    <TreeItem nodeId={'Models'} label={'Models'}></TreeItem>
					<div>
                        {modelList.map(application => (
                            console.log(application),
                            application.map((model) => (
                                <TreeItem nodeId={model.name} label={model.name} onLabelClick={() => setModel(model)}>
                                </TreeItem>
                            ))
                        ))}
                    </div>

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