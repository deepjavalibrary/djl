import React, { useState, useEffect } from "react";
import { makeStyles } from '@material-ui/core/styles';
import Input from '@material-ui/core/Input';
import TextField from '@material-ui/core/TextField';
import TreeView from '@material-ui/lab/TreeView';
import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import ChevronRightIcon from '@material-ui/icons/ChevronRight';
import TreeItem from '@material-ui/lab/TreeItem';
import ModelView from './ModelView';

import {fetchData} from './network.functions'

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
		flex: "0 0 auto",
		alignSelf: "stretch",
		maxHeight: '600px',
		overflowY: "auto",

	},

	model_root: {
		order: 0,
		flex: "2 0 auto",
		alignSelf: "stretch",	},
});








export default function ModelNavigator(props) {


	const classes = useStyles();

	const URL = 'http://'+window.location.hostname+':'+window.location.port+'/modelzoo/models';
	const modelZooData = fetchData(URL);

	const [model, setModel] = useState(null);
	const [modelList, setModelList] = useState(modelZooData);
	const [nameValue, setNameValue] = useState('');
	const [applicationValue, setApplicationValue] = useState('');
	const [versionValue, setVersionValue] = useState('');
	const [versionList, setVersionList] = useState([]);

    const filteredModels =
        Object.keys(modelZooData).map((application) => (
            modelZooData[application].filter((modelReference) => {
                if ((versionValue == '') || (versionValue == modelReference.version)){
                    if ((applicationValue == '') || (applicationValue == application)){
                        return modelReference.name.toLowerCase().includes(nameValue.toLowerCase());
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
                                        {Object.keys(modelZooData).map((application) => (
                                            modelZooData[application].map((modelReference) => (
                                                    handleAdd(modelReference.version)
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
                                        {Object.keys(modelZooData).map((application) => (
                                            <button
                                                disabled={applicationValue == application}
                                                value={application}
                                                onClick={modelApplicationFilterOnChange}
                                            >

                                                {application}

                                            </button>
                                        ))}
                                       <button value='' onClick={modelApplicationFilterOnChange} > Clear </button>
                                    </div>
                                </TreeItem>
                            </TreeView>
                            <button onClick={modelFilterOnChange} value={nameValue} > Search </button>
                        </TreeItem>
                    </TreeView>

				<TreeView

					defaultCollapseIcon={<ExpandMoreIcon />}
					defaultExpandIcon={<ChevronRightIcon />}
				>
                    {nameValue != '' || versionValue!='' || applicationValue!=''
                        ?<div>
                           {
                            modelList.map(application => (
                                application.map((model) => (
                                    <TreeItem nodeId={model.name} label={model.name} onLabelClick={() => setModel(model)}>
                                    </TreeItem>
                                ))
                            ))}
                        </div>
                        :
                          Object.keys(modelZooData).map((application) => (
                            <TreeItem nodeId={application} label={application}>
                                {modelZooData[application].map((model) => (
                                    <TreeItem nodeId={model.name} label={model.name} onLabelClick={() => setModel(model)}>
                                    </TreeItem>
                                ))}
                            </TreeItem>
                          ))
                    }

				</TreeView>
				</div>

				{model != null &&
					<div className={classes.model_root}>
						<ModelView modelRef={model} />
					</div>
				}
			</div>
		</>
	);
}