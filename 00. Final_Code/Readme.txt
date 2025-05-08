RASA SETUP

## 📂 Folder Structure
	rasa_backend/
		- config.yml		
		- domain.yml
		- endpoints.yml

		- `actions/`
         		- action.py
                      
		- `data/`
 			 - nlu.yml
     		 	 - rules.yml
		 	 - stories.yml

		- `models/`
		  - when doing RASA train -- this folder gets created and a model file is placed here after training is success. 

## 📌 How to Run

1. Install required packages given below (if available).

	come to root folder of rasa_backend
	run "python -m venv rasa_env"
	run "rasa_env\Scripts\activate"

	Now run below
	pip install --upgrade pip setuptools wheel
	pip install rasa   (internally pulls required packages - tensorflow, scikit-learn, aiohttp, colorama)


	run "rasa init" (creates sample files (nlu.yml, domain.yml, stories.yml, etc.))
	run "pip install rasa-sdk"

5. Start both rasa actions and rasa shell
	separate command line windows
		rasa run actions
		rasa shell

6. Finally check entries in NLU.yml, domain.yml, stories.yml, rules.yml or use entries in files in GitHub repo. This has to be carefully updated for end to end RASA flow to work. 

FAST API SETUP
## folder - 
  chatbot_api/
    models/
    routers/
    services/
    dbUtils.py
    main.py
    flightBookingOps.py
    config.py

## How to run 
  1. go to chatbot_api folder (e.g. C:\IISc_Capstone\RASA\chatbot_api_with_rasa\chatbot_api)
  2. run the command - uvicorn main:app --reload 

## Dependencies
  pip install fastapi


  
STREAMLIT SETUP
# Dependencies 
  pip install streamlit

# How to run
  1. go to chatbot_api_with_rasa folder 
  2. run command - streamlit run StreamlitInterface.py

OLLAMA SERVER
# Dependencies
  Download Ollama from https://ollama.com
  Install it on your system
#How to run
  1. Go to the folder where Ollama was installed
  2. ollama run mistral

#TWILIO SERVER

#WHATSAPP SERVER
