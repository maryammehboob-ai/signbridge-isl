# SignBridge online: deployment guide (Streamlit Community Cloud)

This folder (`signbridge_deploy`) is a copy of the files the app needs. Your original project is unchanged and the models are byte-identical.
Nothing here has been pushed anywhere. You do steps 1-4 yourself.

## What is in this folder
- `APP/` the Streamlit app (entry file `APP/app.py`); the live page now uses WebRTC (`APP/core/webrtc_session.py`, `APP/views/page_live.py`)
- `MODELS/` isl_landmarks.keras (live model), isl_mobilenetv2.keras (loaded as a fallback), class_names.json
- `SCRIPTS/`, `SIGN_TO_TEXT/`, `SPEECH_TO_TEXT/`, `TEXT_TO_SIGN/` the existing recognition and text code (not modified) and the sign pictures
- `MODEL_EVALUATION/landmarks/` only the saved results and charts for the Model Performance page (no recordings)
- `requirements.txt`, `packages.txt`, `.streamlit/config.toml`, `.streamlit/secrets.toml.example`
- `test_pipeline.py`, `test_pages.py` the tests (run with `.venv_deploy\Scripts\python.exe`)

## 1. Create the GitHub repository (you)
1. Create a new, empty **public** repository on github.com, for example `signbridge-isl` (free Community Cloud needs a public repo).
2. In a terminal, in this folder, set an author that is not your private email. Use the no-reply address shown in
   GitHub > Settings > Emails ("Keep my email addresses private"):
   ```
   git init
   git config user.name "Your Name"
   git config user.email "<id>+<username>@users.noreply.github.com"
   git add .
   git status        # check: no .venv_deploy, no secrets, about 13 MB
   git commit -m "SignBridge online"
   git branch -M main
   git remote add origin https://github.com/<username>/signbridge-isl.git
   git push -u origin main
   ```
   Tell me first if you want me to run any of these.

## 2. Deploy (you, at share.streamlit.io)
1. Sign in with GitHub, click **Create app**, choose the repository, branch `main`, **Main file path: `APP/app.py`**.
2. Open **Advanced settings** and choose **Python 3.12** (TensorFlow 2.19 does not support newer versions).
3. **App URL**: type your wanted name, for example `maryam-signbridge`, so the address is `https://maryam-signbridge.streamlit.app`.
   Names are first come, first served; if it is taken, pick another. It can be changed later under Settings.
4. Click **Deploy**. The first build takes several minutes (TensorFlow is large).

## 3. Optional: TURN relay (only if live video fails on some networks)
Some networks (mobile data, school or office Wi-Fi) block direct live video. A TURN relay fixes that.
Create a free account at a TURN provider, then paste the values into App settings > Secrets using the layout of
`.streamlit/secrets.toml.example`. The app reads them automatically. Never put them in the repository.

## 4. First checks after deploy
Open the URL on a laptop and on a phone: Home loads, Text-to-Sign works, Speech-to-Text (allow the microphone), Model Performance shows 79.1%,
Live Sign-to-Text: press START, allow the camera, show a sign, check the prediction, press Add Letter.

## Known limits
- Reading aloud (text-to-speech) is not available online (it used Windows voices).
- Free tier: limited CPU/memory. At most 3 people can use the live camera at once; others see a "server busy" notice.
- The app sleeps after a period without visitors; the first visit afterwards is slow (models reload).
- Live video needs HTTPS (provided) and a network that allows WebRTC; without a TURN relay a few networks will not connect.
- Not verifiable on this Windows PC: the Linux build on Streamlit Cloud (first deploy is the real test).
