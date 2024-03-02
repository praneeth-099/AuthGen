import React from "react";
import Index from "pages/index";
import ClientLogin from "pages/clientLogin";
import ClientEncodeImage from "pages/clientEncodeImage";
import CustomerDecodeImage from "pages/customerDecodeImage";
import privacyPolicy from "pages/privacyPolicy";
import Page404 from "pages/page404";
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";
import { createGlobalStyle } from "styled-components";
const GlobalStyles = createGlobalStyle`
    body {
        margin: 0;
        padding: 0;
        font-family: sans-serif;
    }
`;
export default () => (
<Router>
<GlobalStyles />
<Switch>
	<Route exact path='/' component={Index}/>
			<Route exact path='/index' component={Index}/>
			<Route exact path='/client-login' component={ClientLogin}/>
			<Route exact path='/client-encode-image' component={ClientEncodeImage}/>
			<Route exact path='/customer-decode-image' component={CustomerDecodeImage}/>
<Route exact path='/privacy-policy' component={privacyPolicy}/>
			<Route component={Page404}/>
</Switch>
</Router>
);
