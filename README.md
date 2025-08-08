# MIMIC-CXR Dicom Viewer

## Purpose: 
To allow clinical collaborators to review MIMIC images and confirm the presence or absence of findings of interest. 

## Guide
At this moment in time, the app requires you to upload a curated CSV. A future version will be deployed directly in Snowflake. 
1. Prepare a CSV file that has columns STUDYID and S3_PATH
2. Open the app by visiting this link in your browser: https://mimicdicomviewer.streamlit.app/
3. Follow the instructions to upload the CSV file from (1). 
4. Browse the images and adjust the windowing as you wish! 