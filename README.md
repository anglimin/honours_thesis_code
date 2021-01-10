# GE4401: Honours Thesis

### Title: Identifying Discrepancies within current transport initiatives: An Urban Informatics Approach

This repository contains the scripts that I have used for my methodology.<br /> 
In sequential order, the scripts are executed as follow: 
  1. reddit.py and twitter.py
  2. topic_modelling.py
  3. Visualisation_Stationarity.py

Following diagram showcases the methodology flow (for pure scripting), as well as, the output and their file names after each run.<br />
![github diagram](https://user-images.githubusercontent.com/58674555/104113414-7c13a880-5334-11eb-86c8-8400baa12336.png)

#### Disclaimer 1: Raw data will not be supplemented in this repository to prevent breach of privacy. Refer to Appendix C to understand data schema of the raw and processed data. <br />
#### Disclaimer 2: reddit.py and twitter.py contain environmental variables that users need to change on their own end
<br />
#### Disclaimer 3: I have also attached the script (<ins>reddit_locations.py</ins>) for identifying locations with Reddit commments through spaCy Named Entity Recognition (NER). However, these results are not utilised for subsequent analysis. Moreover, the trained NER model (spacy_sg) may not be the best identifier of Singapore's locations in lieu of various reasons that will not be covered in this thesis. 

### Initialising the package folder and dependencies.
  1. Git clone the entire package using whatever CLI you are comfortable with
  2. ```console
     pip install -r requirements.txt
     ```
  3. Run the scripts sequentially
