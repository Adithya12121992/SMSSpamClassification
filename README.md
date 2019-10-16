PREREQUISITE
-------------------------
The code in MLIndex.py is written/developed in python3.7 . (Python 3 and above is mandatory)<br/>
Packages required : nltk, sklearn, pandas ,numpy<br/>
Removal of Stopwords and Lemmatization : Download the nltk.download()<br/>

SMS Spam Collection v.1
-------------------------

1. DESCRIPTION
--------------

The SMS Spam Collection is a set of SMS Text messages that have been collected for SMS Spam Classification using Naive Bayes and DT using MaxEnt.The entire set of 5,574 messages is divided into 2 categories, HAM (legitimate) and SPAM.

1.a. STATISTICS
---------------

There is one collection:

- The SMS Spam Collection v.1 (text file: SMSSpamCollection) has a total of 4,179 SMS HAM SMS's (75%) and a total of 1,393 (25%) SPAM SMS's.


1.b. FORMAT
-----------

The files contain one message per line. Each line consists of two columns: one with label (ham or spam) and other with the sample SMS text. Below are some examples:

ham	  Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...<br/>
ham	  Ok lar... Joking wif u oni...<br/>
spam  Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry<br/> question(std txt rate)T&C's apply 08452810075over18's<br/>
ham	  U dun say so early hor... U c already then say...<br/>
ham	  Nah I don't think he goes to usf, he lives around here though<br/>

Note: messages are not sorted in any order.


2. USAGE
--------

We have used the SMS Spam Collection Dataset for Testing and Training our classifier, we have decided to use the Dataset using the split ratio 1:3 (25% for testing and 75% for training).The Full dataset can be download either from : https://www.kaggle.com/uciml/sms-spam-collection-dataset or from the Github repo : https://github.com/Adithya12121992/SMSSpamClassification.git


3. ABOUT
--------

The corpus has been collected by G�mez Hidalgo, J.M., Yamakami, A. Contributions to the Study of SMS Spam Filtering: New Collection and Results. Proceedings of the 2011 ACM Symposium on Document Engineering (DOCENG'11), Mountain View, CA, USA, 2011.

We would like to thank Vivek Chutke (https://www.kaggle.com/vivekchutke/spam-ham-sms-dataset) for making the NUS SMS Corpus available.
