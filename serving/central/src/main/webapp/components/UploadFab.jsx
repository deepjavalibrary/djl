import React, { Component, useState, useEffect, useRef } from "react";
import Backdrop from '@material-ui/core/Backdrop';
import Button from '@material-ui/core/Button';
import Fade from '@material-ui/core/Fade';
import Fab from '@material-ui/core/Fab';
import Modal from '@material-ui/core/Modal';
import TextField from '@material-ui/core/TextField';
import Tooltip from '@material-ui/core/Tooltip';
import AddIcon from '@material-ui/icons/Add';
import ReactDOM from 'react-dom';
import { makeStyles } from '@material-ui/core/styles';
import axios from 'axios'

const useStyles = makeStyles((theme) => ({
      modal: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      },
      paper: {
        backgroundColor: theme.palette.background.paper,
        border: '2px solid #000',
        boxShadow: theme.shadows[5],
        padding: theme.spacing(2, 4, 3),
      },
    }));





export default function UploadFab() {
    const classes = useStyles();
    const [show, setShow] = useState(false);
    const [selectedFile, setSelectedFile] = useState(null);
    const [data, setData] = useState([]);

    const handleShow = () => setShow(true);

    const onChangeHandler=(event)=>{
        setSelectedFile(event.target.files[0])
        console.log(event.target.files[0])
    }

    const handleClose = (event) => {
           setShow(false);
           alert("Your file is being uploaded!")
           axios.get("http://"+window.location.host+"/uploading/models?modelName="+selectedFile.name)
                .then(function(response) {
                    let appdata = Object.keys(response.data).map(function(key) {
                        return {
                            key: key,
                            name: response.data[key]
                        };
                    });
                    setData(appdata);
                    console.log(appdata)
                })
        };

	function handleFabClick() {
        handleShow()
        console.log("Fab Button has been pressed.")
        }

    return (
    		<>
    		    <div>
                    <Tooltip title="Press to add a Model" aria-label="add">
                        <Fab
                        color="primary"
                        onClick={() => handleFabClick() }
                        >
                            <AddIcon />
                        </Fab>
                    </Tooltip>

                    <Modal
                        aria-labelledby="transition-modal-title"
                        aria-describedby="transition-modal-description"
                        className={classes.modal}
                        open={show}
                        closeAfterTransition
                        BackdropComponent={Backdrop}
                        BackdropProps={{
                          timeout: 500,
                        }}
                      >
                            <Fade in={show}>
                              <div className={classes.paper}>
                                <h2 id="transition-modal-title">Model Uploader</h2>
                                     <input type="file" name="file" onChange={onChangeHandler}/>
                                    <button type="button" onClick={handleClose}>
                                        press to close
                                    </button>
                              </div>
                            </Fade>
                    </Modal>
                </div>
    		</>
    );
}