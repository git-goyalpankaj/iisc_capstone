# Go to "rasa-chatbot-iisc"  repo under GitHub Project Repository

This repository contains the implementation of a RASA actions and RASA NLU+core definition. 

## IMPORTANT ##
Only 1 Workflow is defined for 1 intent only - "Upgrade_Request".
Follow below folders and files under them. 

## 📂 Folder Structure

rasa-chatbot-iisc/

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