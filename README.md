# BTP-WebApp

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

1. Install docker from [here](https://docs.docker.com/engine/install/).
2. Install docker-compose.
3. Navigate to the backend directory and run
```
mkdir asr_models
mkdir audio_files
```
4. Download deepspeech pretrained model and scorer and place them in asr_models. 
<br>
Model: https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
<br>
Scorer: https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer


5. Run the following commands to start the server.
```
docker-compose -f docker-compose.yaml up
conda env create -f environment.yml
uvicorn main:app --reload
```
