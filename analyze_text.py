# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 14:55:52 2019

@author: rmuh
"""
import urllib.request
from bs4 import BeautifulSoup
import nltk
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

url = urllib.request.urlopen('https://hernancasciari.com/blog/cupido-motorizado/') 
html = url.read()

#Cleaning the text
soup = BeautifulSoup(html,"html5lib")

#Keep only the main text (without the comments)
main = soup.find('section', class_="medium-font-size section-single-content")
 
text = main.get_text(strip=True)

#Tokenize the text
tokens = re.findall(r"[\w']+", text)
print(tokens)

tokens_wsw = tokens.copy()

#Delete stopwords
sw = stopwords.words('spanish')

sw.extend(['Y','iba','iban','La','Lo','Si','si','di','sos','Que','Qué','Se',
           'No','A','Me','Ni','Estoy','El','Yo','Mi','Esa','Ya','va','Sobre',
           'sobre','ver'])
 
for token in tokens:
    if token in sw:
        tokens_wsw.remove(token)
 
#Compute the frequency     
freq = nltk.FreqDist(tokens_wsw)
     
plt.figure(figsize = (12,8))
freq.plot(25,cumulative=False)
 