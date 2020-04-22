# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 21:56:53 2019

@author: Saad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


bankdata2 = pd.read_csv("trainingCommonVoicePreProcessedWithAgeGroupsRemovingOther.csv")

count_gender =pd.value_counts(bankdata2['gender'])
total_gender =sum(count_gender)

count_age =pd.value_counts(bankdata2['age_group'])
total_age = sum(count_age)

count_acc =pd.value_counts(bankdata2['accent'])
total_acc = sum(count_acc)

count_gender.plot(kind='bar',rot=0)
count_age.plot(kind='bar',rot=0)
count_acc.plot(kind='bar',rot=0)
