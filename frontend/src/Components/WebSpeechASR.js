import axios from "axios";
import Grid from '@mui/material/Grid';
import KeyboardVoiceIcon from '@mui/icons-material/KeyboardVoice';
import IconButton from '@mui/material/IconButton';

const WebSpeechASR = ({ setQueryText, setQueryResponse }) => {
    let SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition
    let asr = new SpeechRecognition()
    asr.lang = 'en-IN'
    asr.continuous = false
    asr.interimResults = false
    asr.maxAlternatives = 1


    function startAsr(setQueryText, setQueryResponse) {
        asr.start()
        asr.onresult = (event) => {
            let transcript = event.results[0][0].transcript
            setQueryText(transcript)
            console.log(transcript)
            axios.post('http://localhost:8000/detect_intent',
                {
                    "query_text": transcript
                })
                .then(function (response) {
                    console.log(response)
                    setQueryResponse(response.data['intent'])
                })
                .catch(function (error) {
                    console.log(error)
                })
        }
    }

    return (
        <Grid item xs={12} sm={12} md={12} lg={12} xl={12}>
            <IconButton
                aria-label="record"
                size="large"
                color="primary"
                onClick={(e) => startAsr(setQueryText, setQueryResponse)}>
                <KeyboardVoiceIcon fontSize="inherit" />
            </IconButton>
        </Grid>
    )
}

export default WebSpeechASR