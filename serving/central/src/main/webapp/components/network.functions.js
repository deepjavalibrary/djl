
import axios from 'axios'
import React, { useState, useEffect } from "react";

export const fetchData = (url ) => {
		const [data, setData] = useState([]);
	
		// empty array as second argument equivalent to componentDidMount
		useEffect(() => {
			async function fetchData() {
			console.log("fetch data from "+url)
				axios.get(url)
					.then(function(response) {
						console.log(response);
						setData(response.data)
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