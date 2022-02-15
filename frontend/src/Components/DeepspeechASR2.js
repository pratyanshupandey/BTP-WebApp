import axios from 'axios';
import {RecordRTC, StereoAudioRecorder, invokeSaveAsDialog} from 'recordrtc'
import {useEffect, useState} from "react";

const DeepspeechASR2 = ({setQueryText, setQueryResponse}) => {

    const [recorder, setRecorder] = useState("")

    useEffect(() => {
        navigator.mediaDevices.getUserMedia({
            audio: true
        }).then(async function (stream) {
            var recorder = new StereoAudioRecorder(stream, {
                // type: 'audio',
                // mimeType: 'audio/wav',
                // recorderType: new StereoAudioRecorder,

                // used by StereoAudioRecorder
                // the range 22050 to 96000.
                sampleRate: 44100,

                // used by StereoAudioRecorder
                // the range 22050 to 96000.
                // let us force 16khz recording:
                // desiredSampRate: 16000,

                // used by StereoAudioRecorder
                // Legal values are (256, 512, 1024, 2048, 4096, 8192, 16384).
                bufferSize: 16384,

                // used by StereoAudioRecorder
                // 1 or 2
                numberOfAudioChannels: 2

            })
            setRecorder(recorder)
        })

    }, [])

    function stopRecording() {
        recorder.stop(function (blob) {
            // let blob = recorder.getBlob();
            console.log("Stopped")
            invokeSaveAsDialog(blob);
        })
    }

    function startRecording() {
        recorder.record()
    }

    return (
        <div>
            <button onClick={startRecording}>Start</button>
            <button onClick={stopRecording}>Stop</button>
        </div>)
}

export default DeepspeechASR2
