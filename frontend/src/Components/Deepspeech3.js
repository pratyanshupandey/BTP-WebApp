import axios from 'axios';
import { useEffect, useState } from "react";
import MediaStreamRecorder from "msr";
import { Grid } from '@mui/material';
import KeyboardVoiceIcon from '@mui/icons-material/KeyboardVoice';
import StopCircleOutlinedIcon from '@mui/icons-material/StopCircleOutlined';
import { Button } from '@mui/material';
import { Box } from '@mui/system';

const Deepspeech3 = ({ setQueryText, setQueryResponse }) => {

    const [recorder, setRecorder] = useState("")

    useEffect(() => {
        var mediaConstraints = {
            audio: true
        };

        navigator.getUserMedia(mediaConstraints, onMediaSuccess, onMediaError);

        function onMediaSuccess(stream) {
            var mediaRecorder = new MediaStreamRecorder(stream);
            // mediaRecorder.recorderType = StereoAudioRecorder;
            mediaRecorder.mimeType = 'audio/wav'; // check this line for audio/wav
            // mediaRecorder.ondataavailable = function (blob) {
            //     // POST/PUT "Blob" using FormData/XHR2
            //     // var blobURL = URL.createObjectURL(blob);
            //     // document.write('<a href="' + blobURL + '">' + blobURL + '</a>');
            //     console.log("Saving")
            //     invokeSaveAsDialog(blob)
            //     var file = new File([blob], "recording.wavrea")
            // };
            mediaRecorder.ondataavailable = function (typedArray) {
                let dataBlob = new Blob([typedArray], { type: 'audio/wav' });
                let file = new File([dataBlob], "recording.wav")
                const formData = new FormData();
                formData.append('audio_file', file)
                axios({
                    method: "post",
                    url: "http://localhost:8000/audio_upload",
                    data: formData,
                    headers: {
                        "content-type": `multipart/form-data;`
                    }
                }).then(function (response) {
                    console.log(response)
                    setQueryResponse(response.data['intent'])
                    setQueryText(response.data['transcript'])

                }).catch(function (error) {
                    console.log(error)
                })
            }
            setRecorder(mediaRecorder)
        }

        function onMediaError(e) {
            console.error('media error', e);
        }

    }, [])

    function stopRecording() {
        console.log("Stopping")
        recorder.stop()
        console.log("Stopped")
    }

    function startRecording() {
        recorder.start(10000)
        console.log("Started")
    }


    return (
        <Grid item xs={12} sm={12} md={12} lg={12} xl={12}>
            <Button
                variant="contained"
                color='primary'
                startIcon={<KeyboardVoiceIcon />}
                onClick={startRecording}>
                Start Recording
            </Button>
            
            <Button
                variant="contained"
                color='warning'
                startIcon={<StopCircleOutlinedIcon />}
                onClick={stopRecording}>
                Stop Recording
            </Button>
        </Grid>
    )
}

export default Deepspeech3
