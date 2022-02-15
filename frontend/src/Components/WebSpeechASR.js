const axios = require('axios')

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

    // if(!isWebSpeechUsable)
    //     return (
    //         <div>Your browser does not support Web Speech API.</div>
    //     )

    return (
        <div>
            <h1>Recorder</h1>
            <button onClick={(e) => startAsr(setQueryText, setQueryResponse)}>Record</button>
        </div>
    )
}

export default WebSpeechASR