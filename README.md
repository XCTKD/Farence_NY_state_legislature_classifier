# Farence_NY_state_legislature_classifier
 
The data used in this project was scraped from the New York state senate website. This batch of data contains over 7,000 records - each records contains the following information:
- Bill Number
- URL
-Bill Title
- Name of sponsoring senator
- District of sponsoring senator
- Number of aye votes
- Number of nay votes
- Date of last legislative action
- Date delivered to gov (if applicable)
- Boolean for sign/vetoed
- Date signed/vetoed
- Sponsor memo (two parts)
- Text of the bill

My goal was to build a classifer and feed it information from the 2013 legislative session so it could learn how to predict if a bill was signed based on it's language.

I think I got mostly there. I have data for the 2011, 2015, 2017, and 2019 sessions I didn't yet get a chance to feed it before tonight's deadline.

But the classifier it seemed was able to predict fairly well if a bill got signed.
