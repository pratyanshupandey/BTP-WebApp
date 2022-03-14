import { useState } from "react";
import Box from '@mui/material/Box';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormHelperText from '@mui/material/FormHelperText';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';
import FlightImage from "../assets/flight.jpg"
import WebSpeechASR from "./WebSpeechASR";
import Deepspeech3 from "./Deepspeech3";
import Button from '@mui/material/Button';
import Grid from '@mui/material/Grid';
import axios from "axios";
import { Input } from "@mui/material";
import TextField from '@mui/material/TextField';
import { Paper } from "@mui/material";


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
    const [queryResponse, setQueryResponse] = useState("Flight")

    let SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition
    let isWebSpeechUsable = isChromeEdge() && (SpeechRecognition !== undefined)

    const [useWebSpeech, setUseWebSpeech] = useState(isWebSpeechUsable)

    function onSubmit(e) {
        e.preventDefault()
        console.log("query submitted")
        console.log(queryText)
        axios.post('http://localhost:8000/detect_intent',
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


    const domains = ["Flight"]
    const background_images = {
        Flight: FlightImage
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
            backgroundSize: 'cover'
        }}>

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
                            <Grid item xs={6} sm={6} md={8} lg={6} xl={6} alignItems={'left'}>
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
                                    <FormHelperText>Select your domain</FormHelperText>
                                </FormControl>
                            </Grid>
                            <Grid item xs={6} sm={6} md={8} lg={6} xl={6}>
                                <Button variant="contained" onClick={(e) => setUseWebSpeech(!useWebSpeech)}>{useWebSpeech ? "Use Deepspeech" : "Use WebSpeech"}</Button>
                            </Grid>

                            {useWebSpeech ? <WebSpeechASR setQueryText={setQueryText} setQueryResponse={setQueryResponse} />
                                : <Deepspeech3 setQueryText={setQueryText} setQueryResponse={setQueryResponse} />}

                            <Grid item xs={12} sm={12} md={12} lg={12} xl={12}>


                                <form onSubmit={onSubmit}>
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
                                        <Grid item xs={12} sm={12} md={12} lg={2} xl={12}>
                                            <Input type="submit" />
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