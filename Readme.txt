System Description

Dataset - ml-latest-small, poster url from IMDB 
Frontend - Vue, Axios 
Backend - Fastapi
Recommend algorithm - user-based centered-KNN, content based one hot vector

Cold start stage - randomly select movies to get user profile
1st round recommend - randomly choose a recommend algorithm
2nd round recommend - based on user profile and 1st feedback
Evaluation -  paired t-Test and two-sample t-Test


how to run: same as demo 
use Chrome and install the apps named “web server for chrome”, and choose this folder
-pip install –r requirements.txt
-uvicorn main:app