import {useState} from "react";
import WebSpeechASR from "./WebSpeechASR";
import DeepSpeechASR from "./DeepSpeechASR";
import DeepspeechASR2 from "./DeepspeechASR2";
import Deepspeech3 from "./Deepspeech3";
import axios from "axios";


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

const DemoPage = () => {
    const [queryText, setQueryText] = useState("")
    const [queryResponse, setQueryResponse] = useState("")

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


    return (
        <div>
            <button onClick={(e) => setUseWebSpeech(!useWebSpeech)}>{useWebSpeech ? "Use Deepspeech" : "Use WebSpeech"}</button>
            {useWebSpeech ? <WebSpeechASR setQueryText={setQueryText} setQueryResponse={setQueryResponse}/>
                : <Deepspeech3 setQueryText={setQueryText} setQueryResponse={setQueryResponse}/>}
            <form onSubmit={onSubmit}>
                <input type={"text"} value={queryText} onChange={(e) => setQueryText(e.target.value)}/>
                <input type={"submit"}/>
            </form>
            <p>{queryResponse}</p>
        </div>
    )
}

export default DemoPage