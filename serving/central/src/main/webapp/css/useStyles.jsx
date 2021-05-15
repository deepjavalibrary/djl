import { makeStyles } from '@material-ui/core/styles';

export const theme = {
	backgroundColor: '#1D2731',
	titleColor: "#328cc1",
	overlayColor: '#0B3C5D',
	textColor: '#ffffff',
	dividerColor: 'rgba(228, 168, 64 , 1)',
};



const useStyles = makeStyles({
	appRoot: {
		// background: "rgb(14,14,13)",
		background: theme.backgroundColor,
		display: 'flex',
		flexDirection: 'column',
		minHeight: '100vh',
		color: '#ffffff',
	},


	content: {
		flexGrow: 1,
	},



	pageTitle: {
		color: "#ffffff",
		bottom: '4rem',
		right: '2rem',
		zIndex: '100',
		position: 'fixed',
		'-webkit-box-shadow': '0px 10px 13px -7px #000000, 5px 5px 15px 5px rgba(11,60,93,0)',
		boxShadow: '0px 10px 13px -7px #000000, 5px 5px 15px 5px rgba(11,60,93,0)',
	},
});




export default useStyles;