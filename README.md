# BTP-WebApp

The web app is deployed at https://asr.iiit.ac.in/Intent_Detection_App/

<br />

The server is deployed using nginx. It is registered as a service by the name of intent_detection. This allows the server to restart itself in case of failure or when the system reboots. Thus it now works as any other service and can be interacted with using systemctl.

<br />

## Managing the existing service for the server

To check the status of the running server
```
systemctl status intent_detection.service
```

To disable the server from starting automatically on startup of VM
```
systemctl disable intent_detection.service
```

To enable the server to start automatically on startup of VM
ON by default
```
systemctl enable intent_detection.service
```

to manually start the server
```
systemctl start intent_detection.service
```

to manually stop the server
```
systemctl stop intent_detection.service
```
<br />
<br />

# Running the code locally

## Running the frontend
1. Run
```
cd frontend
npm install
npm start
```
2. Build the production build and serve it locally using 
```
npm run build
npm install serve
npx serve -g build
```

<br>

<hr>

<br>

## Instructions to run the backend

1. Create the environment according to the provided environment.yml file. Install OpenAI Whisper.
2. Navigate to the directory and place pretrained models for whisper and intent detection into the corresponding folders.
3. Run the following commands to start the server.
```
uvicorn main:app
```
