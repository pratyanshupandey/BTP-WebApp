import { useState } from "react";
import Box from '@mui/material/Box';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';
import FlightImage from "../assets/flight3.jpeg"
import AssistantImage from "../assets/assistant.jpg"
import WebSpeechASR from "./WebSpeechASR";
import OpenAIWhisper from "./OpenAIWhisper";
import Button from '@mui/material/Button';
import Grid from '@mui/material/Grid';
import axios from "axios";
import TextField from '@mui/material/TextField';
import { Paper } from "@mui/material";
import Carousel from 'react-material-ui-carousel'
import Fab from '@mui/material/Fab';
import HelpIcon from '@mui/icons-material/Help';
import Dialog from '@mui/material/Dialog';
import { Image } from "mui-image";
import LayoutHelpImg from "../assets/layout_help.png"
import WebspeechHelpImg from "../assets/webspeech_help.png"
import DeepspeechHelpImg from "../assets/deepspeech_help.png"

function isChromeEdge() {
    const isGoogleBot = navigator.userAgent.toLowerCase().indexOf('googlebot') !== -1
    const isEdge = !!window.StyleMedia
    const isFirefox = typeof InstallTrigger !== 'undefined'
    const isOpera = (!!window.opr && !!window.opr.addons) || !!window.opera
        || navigator.userAgent.indexOf(' OPR/') >= 0
    const isChrome = !isGoogleBot && !isEdge && !isOpera && !!window.chrome && (
        !!window.chrome.webstore
        || navigator.vendor.toLowerCase().indexOf('google inc.') !== -1
    )
    return isChrome || isEdge
}


const ModelPage = () => {

    const [queryText, setQueryText] = useState("")
    const [queryResponse, setQueryResponse] = useState("response")
    const [openHelp, setOpenHelp] = useState(false)


    let SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition
    let isWebSpeechUsable = isChromeEdge() && (SpeechRecognition !== undefined)

    const [useWebSpeech, setUseWebSpeech] = useState(isWebSpeechUsable)

    function onSubmit(e) {
        e.preventDefault()
        console.log("query submitted")
        console.log(queryText)
        axios.post('https://asr.iiit.ac.in/intent_detection/detect_intent/?domain=' + domain,
            {
                "query_text": queryText
            })
            .then(function (response) {
                console.log(response)
                setQueryResponse(response.data['intent'])
            })
            .catch(function (error) {
                console.log(error)
            })
    }


    const domains = ["Assistant_amazon", "Flight_atis", "Assistant_snips"]
    const background_images = {
        Assistant_amazon: AssistantImage,
        Flight_atis: FlightImage,
        Assistant_snips: AssistantImage,
    }

    const [domain, setDomain] = useState(domains[0]);


    const handleChange = (event) => {
        setDomain(event.target.value);
    };

    return (
        <div style={{
            backgroundImage: `url(${background_images[domain]})`,
            backgroundPosition: 'center',
            backgroundRepeat: "no-repeat",
            backgroundSize: 'cover',
            minHeight: 700
        }}>

            <Fab
                color="secondary"
                aria-label="edit"
                onClick={e => setOpenHelp(!openHelp)}
                sx={{
                    margin: 0,
                    top: 'auto',
                    right: 20,
                    bottom: 20,
                    left: 'auto',
                    position: 'fixed'
                }}>
                <HelpIcon />
            </Fab>

            <Dialog open={openHelp} onClose={e => setOpenHelp(false)}
                fullwidth={true}
                maxWidth={false}
                maxHeight={false}
                sx={{ maxWidth: 1400, minHeight: 700 }}>
                <Carousel
                    sx={{ width: 1032, height: 465 }}
                    autoPlay={false}
                    navButtonsAlwaysVisible='true'
                    animation="slide"
                    indicators={true}>
                    <Image
                        src={LayoutHelpImg}
                        height='310'
                        width='688'
                        // fit="contain"
                        duration={0}
                    />
                    <Image
                        src={WebspeechHelpImg}
                        height='310'
                        width='688'
                        // fit="contain"
                        duration={0}
                    />
                    <Image
                        src={DeepspeechHelpImg}
                        height='310'
                        width='688'
                        // fit="contain"
                        duration={0}
                    />

                </Carousel>

            </Dialog>



            <Grid container rowSpacing={5}>
                <Grid item xs={12} sm={12} md={12} lg={2.5} xl={2}></Grid>
                <Grid item xs={12} sm={12} md={12} lg={7} xl={8}>
                    <Box sx={{
                        flexGrow: 1,
                        padding: 2,
                        margin: 4,
                        display: "flex",
                        border: '1px solid grey',
                        alignItems: 'center',
                        bgcolor: "azure"
                    }}>


                        <Grid container rowSpacing={5}>
                            <Grid item xs={12} sm={12} md={4} lg={4} xl={4} sx={{ display: "flex", justifyContent: "flex-start" }}>
                                <FormControl sx={{ m: 1, minWidth: 120 }}>
                                    <InputLabel id="domain-select">Domain</InputLabel>
                                    <Select
                                        labelId="domain-select-helper-label"
                                        id="domain-select-helper"
                                        value={domain}
                                        label="Domain"
                                        onChange={handleChange}
                                    >
                                        {domains.map(domain => <MenuItem key={domain} value={domain}>{domain}</MenuItem>)}
                                    </Select>
                                    {/* <FormHelperText>Select your domain</FormHelperText> */}
                                </FormControl>
                            </Grid>
                            <Grid item xs={6} sm={6} md={4} lg={4} xl={4} sx={{ display: "flex", alignItems: 'center' }}>
                                Currently using {!useWebSpeech ? "OpenAI Whisper" : "WebSpeech"}

                            </Grid>

                            <Grid item xs={6} sm={6} md={4} lg={4} xl={4} sx={{ display: "flex", alignItems: 'center', justifyContent: 'right' }}>
                                <Button variant="contained" onClick={(e) => setUseWebSpeech(!useWebSpeech)}>{useWebSpeech ? "Use OpenAI Whisper" : "Use WebSpeech"}</Button>
                            </Grid>

                            {useWebSpeech ? <WebSpeechASR setQueryText={setQueryText} setQueryResponse={setQueryResponse} domain={domain} />
                                : <OpenAIWhisper setQueryText={setQueryText} setQueryResponse={setQueryResponse} domain={domain} />}

                            <Grid item xs={12} sm={12} md={12} lg={12} xl={12}>


                                <form>
                                    <Grid container>
                                        <Grid item xs={12} sm={12} md={12} lg={10} xl={12}>
                                            <TextField
                                                required
                                                fullWidth
                                                id="query-text"
                                                label="Required"
                                                value={queryText}
                                                onChange={(e) => setQueryText(e.target.value)}
                                            />
                                        </Grid>
                                        <Grid item xs={12} sm={12} md={12} lg={2} xl={12} sx={{ display: "flex", alignItems: 'center', justifyContent: 'center' }}>
                                            <Button variant="contained" onClick={onSubmit}>Submit</Button>
                                            {/* <Input type="submit" backgroundColor='green'/> */}
                                        </Grid>
                                    </Grid>

                                </form>
                            </Grid>

                            <Grid item xs={12} sm={12} md={12} lg={12} xl={12}>
                                <Paper elevation={3} sx={{ minHeight: 150 }}>
                                    {queryResponse}
                                </Paper>
                            </Grid>

                        </Grid>
                    </Box>
                </Grid>
                <Grid item xs={12} sm={12} md={12} lg={2.5} xl={2}></Grid>
            </Grid>


        </div>
    )
}

export default ModelPage

// When using MUI Grid API, it is important to notice that when direction=“column”, alignItems affects horizontal alignment and justifyContent affects vertical alignment. When direction=“row” (the default), the opposite is true.