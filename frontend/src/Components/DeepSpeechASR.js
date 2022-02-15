import axios from "axios"
import Recorder from "opus-recorder"

const DeepSpeechASR = ({setQueryText, setQueryResponse}) => {


    var rec = new Recorder({
        numberOfChannels: 2,
        wavBitDepth: 16
    })
    if (!Recorder.isRecordingSupported())
        return (<div>Recording not supported</div>)

    rec.ondataavailable = function (typedArray) {
        let dataBlob = new Blob([typedArray], {type: 'audio/wav'});
        let file = new File([dataBlob], "recording.wav")
        const formData = new FormData();
        formData.append('files.file', file)
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

    return (
        <div>
            <button onClick={(e) => rec.start().catch(function (e) {
                console.log('Error encountered:', e.message)
            })}>Start
            </button>
            <button onClick={(e) => rec.stop()}>Stop</button>
        </div>
    )
}
export default DeepSpeechASR