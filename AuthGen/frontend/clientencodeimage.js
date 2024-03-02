importReact, {useState} from"react";
importthemefrom"theme";
import { Theme, Image, Text, LinkBox, Box, Section, Button } from"@quarkly/widgets";
import { GlobalQuarklyPageStyles } from"global-page-styles";
import { RawHtml, Override } from"@quarkly/components";
import*asComponentsfrom"components";
import { FiMenu } from"react-icons/fi";
import { AiOutlineMenu } from"react-icons/ai";
import { FileUploader } from"react-drag-drop-files";
import { ColorRing } from'react-loader-spinner';

exportdefault (() => {

    
    const [isLoggedIn,set_isLoggedIn] =useState(sessionStorage.getItem('isLoggedIn'));

    
    constfileTypes= ["JPEG", "PNG", "JPG"];

    const [File, setFile] =useState(null);
    const [Loading,setLoading] =useState(null);

    consthandleChange= (file) => {
        setFile(file[0]);   
    };

    constencode=async (event) => {
        event.preventDefault();

        constformData=newFormData();
        formData.append('file', File);
        formData.append('secret',sessionStorage.getItem('secret'));

        try {
            setLoading(true);
            constresponse=awaitfetch('http://localhost:7500/make_image', {
              method:'POST',
              body:formData,
            });
            
            if (response.ok) {
              constblob=awaitresponse.blob();
              consturl=window.URL.createObjectURL(blob);
      
              constlink=document.createElement('a');
              link.href=url;
      
              link.download=File.name.split('.')[0]+'_encoded.jpg';
              document.body.appendChild(link);
              
              link.click();
            
              setLoading(false);
      
            } else {
              console.error('Failed to fetch image');
            }
            setLoading(false);
          } catch (error) {
            setLoading(false);
            console.error('Error:', error);
        }

    }

    if (isLoggedIn){
        return<Themetheme={theme}>
        <head>
            <title>Encrypt</title>
        </head>
        <GlobalQuarklyPageStylespageUrl={"client-encode-image"}/>
        <Sectionsm-padding="8px 0 8px 0"quarkly-title="Header-4"background="#EDF2F6">
            <Overrideslot="SectionContent"flex-direction="row"justify-content="space-between"/>
            <Box
                display="flex"
                padding="12px 0"
                justify-content="space-between"
                align-items="flex-start"
                flex-direction="row"
                width="30%"
                sm-width="50%"
                sm-align-items="center"
                sm-flex-direction="row"
                sm-justify-content="flex-start"
                md-width="50%"
                lg-width="70%"
                md-justify-content="flex-start"
            >
                <LinkBoxflex-direction="row"href="/index"display="flex"grid-gap="12px">
                    <Imagesrc="https://uploads.quarkly.io/6457b971d28fe100213a0f35/images/flower-logo-template.svg?v=2023-08-06T19:34:51.878Z"display="block"width="36px"height="36px"/>
                    <Text
                        margin="0"
                        md-margin="0px 0 0px 0"
                        text-align="left"
                        font="--headline3"
                        sm-margin="0px 0 0px 0"
                        display="block"
                    >
                        AuthGen
                    </Text>
                </LinkBox>
            </Box>
            <Components.QuarklycommunityKitMobileSidePanel
                menuPosition="full"
                breakpoint="lg"
                width="70%"
                sm-width="50%"
                md-width="50%"
                lg-width="30%"
            >
                <Overrideslot="Children"md-display="flex"/>
                <Override
                    slot="Content"
                    padding="0px 0px 0px 0px"
                    background="rgba(255, 255, 255, 0)"
                    lg-background="#ffffff"
                    lg-margin="0px 0px 0px 0px"
                />
                <Override
                    slot="Button Text"
                    font="normal 600 16px/1.5 -apple-system, system-ui, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif"
                    text-transform="uppercase"
                    letter-spacing="1px"
                    sm-font="normal 600 14px/1.5 -apple-system, system-ui, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif"
                    sm-margin="0px 2px 0px 0px"
                    lg-margin="0px 0px 0px 0px"
                />
                <Override
                    slot="Button Icon :closed"
                    category="fi"
                    icon={FiMenu}
                    size="32px"
                    padding="5px 7px 5px 7px"
                    border-radius="50px"
                />
                <Override
                    slot="Button Icon"
                    width="28px"
                    height="28px"
                    category="ai"
                    icon={AiOutlineMenu}
                    color="--dark"
                    size="24px"
                    lg-width="32px"
                    lg-height="32px"
                />
                <Override
                    slot="Cross"
                    lg-width="32px"
                    lg-height="32px"
                    size="32px"
                    top="24px"
                    right="24px"
                />
                <Box
                    align-items="center"
                    lg-justify-content="center"
                    lg-align-items="flex-start"
                    justify-content="flex-end"
                    display="flex"
                    lg-flex-direction="column"
                    lg-margin="0px auto 0px auto"
                    lg-min-width="300px"
                    lg-max-width="1280px"
                    lg-width="90%"
                    lg-padding="24px 0px 48px 0px"
                >
                    <Box
                        display="none"
                        lg-width="100%"
                        lg-margin="0px 0px 24px 0px"
                        lg-display="flex"
                        lg-padding="12px 0px 12px 0px"
                    >
                        <LinkBoxflex-direction="row"href="/index"display="flex"grid-gap="12px">
                            <Imagesrc="https://uploads.quarkly.io/6457b971d28fe100213a0f35/images/flower-logo-template.svg?v=2023-08-06T19:34:51.878Z"display="block"width="36px"height="36px"/>
                            <Text
                                margin="0"
                                md-margin="0px 0 0px 0"
                                text-align="left"
                                font="--lead"
                                sm-margin="0px 0 0px 0"
                                display="block"
                                lg-font="--headline3"
                            >
                                Company
                            </Text>
                        </LinkBox>
                    </Box>
                    <Components.QuarklycommunityKitMenu
                        display="flex"
                        filterMode="exclude"
                        filterPages="/index"
                        grid-gap="8px"
                        lg-flex-direction="column"
                        lg-padding="6px 0px 6px 0px"
                        lg-margin="0px 0px 24px 0px"
                        align-items="center"
                        flex-wrap="wrap"
                        overflow-x="visible"
                        overflow-y="visible"
                        lg-align-items="flex-start"
                    >
                        <Override
                            slot="link"
                            color="--darkL2"
                            hover-color="--primary"
                            font="--lead"
                            text-decoration-line="initial"
                            transition="color 0.1s ease 0s"
                            lg-font="--lead"
                        />
                        <Override
                            slot="item-active"
                            border-width="0 0 2px 0"
                            border-style="solid"
                            border-color="--color-darkL2"
                            lg-border-width="0 0 0 2px"
                        />
                        <Overrideslot="item"padding="6px 8px 6px 8px"lg-padding="6px 12px 6px 12px"/>
                        <Overrideslot="link-active"cursor="default"color="--darkL2"hover-color="--darkL2"/>
                        <Overrideslot="link-about"/>
                    </Components.QuarklycommunityKitMenu>
                    <Box
                        width="25%"
                        display="none"
                        lg-width="100%"
                        lg-flex-direction="column"
                        lg-align-items="flex-start"
                        lg-display="flex"
                        justify-content="space-around"
                        align-items="center"
                        flex-wrap="wrap"
                        lg-margin="0px 0px 0px 0px"
                    >
                    </Box>
                </Box>
            </Components.QuarklycommunityKitMobileSidePanel>
        </Section>
        <Sectionsm-padding="8px 0 8px 0"quarkly-title="Header-5"background="#EDF2F6"min-height="85vh"align-items="center">
        <Boxalign-items="center"display="flex"justify-content="center"flex-direction="column">
        
        <FileUploader
            multiple={true}
            handleChange={handleChange}
            name="file"
            types={fileTypes}
        />
        <Textwidth="100%"
                    lg-width="100%"
                    font="--lead"
                    color="--grey"
                    margin="10px 0px 10px 0px"
                    text-align="center">{File ? `File name: ${File.name}` : "No files uploaded"}</Text>

        {File&&<Buttonpadding="11px 24px 11px 24px"
                                font="normal 400 20px/1.5 --fontFamily-sans"
                                sm-margin="10px 0px 0px 0px"
                                sm-width="100%"
                                focus-box-shadow="none"
                                color="--light"
                                background="#1A1B41"
                                border-radius="8px"
                                hover-background="--color-darkL1"
                                hover-color="--light"
                                hover-transition="background-color 0.5s ease 0s"
                                transition="background-color 0.1s ease 0s"
                                margin="10px 0px 15px 0px"
                                hover-border-width="0px"
                                hover-border-style="none"
                                className="signup"
                                id="signup-action"
                                onClick = {encode}>Encode</Button>}
            {Loading ? (<ColorRing
            visible={true}
            height="80"
            width="80"
            ariaLabel="blocks-loading"
            wrapperStyle={{}}
            wrapperClass="blocks-wrapper"
            colors={['#e15b64', '#f47e60', '#f8b26a', '#abbd81', '#849b87']}
        />) : (<></>)}
        </Box>
        </Section>
        <RawHtml>
            <styleplace={"endOfHead"}rawKey={"6575b98c1d712a00232c66dc"}>
                {":root {\n  box-sizing: border-box;\n}\n\n* {\n  box-sizing: inherit;\n}"}
            </style>
        </RawHtml>
        </Theme>;
    }
    else{
        set_isLoggedIn(sessionStorage.getItem('isLoggedIn'));
        window.location.href='/client-login';
        return ;
    }
    
});
