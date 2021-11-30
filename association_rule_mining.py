#-------------------------------------------------------------------------
# AUTHOR: Vincent Verdugo
# FILENAME: association_rule_mining.py
# SPECIFICATION: description of the program
# FOR: CS 4210 - Assignment #5
# TIME SPENT: 1 hour 20 mins
#-----------------------------------------------------------*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

#Use the command: "pip install mlxtend" on your terminal to install the mlxtend library

#read the dataset using pandas
df = pd.read_csv('retail_dataset.csv', sep=',')

#find the unique items all over the data an store them in the set below
itemset = set()
for i in range(0, len(df.columns)):
    items = (df[str(i)].unique())
    itemset = itemset.union(set(items))

#remove nan (empty) values by using:
itemset.remove(np.nan)

#To make use of the apriori module given by mlxtend library, we need to convert the dataset accordingly. Apriori module requires a
# dataframe that has either 0 and 1 or True and False as data.
#Example:

#Bread Wine Eggs
#1     0    1
#0     1    1
#1     1    1

#To do that, create a dictionary (labels) for each transaction, store the corresponding values for each item (e.g., {'Bread': 0, 'Milk': 1}) in that transaction,
#and when is completed, append the dictionary to the list encoded_vals below (this is done for each transaction)
#-->add your python code below

encoded_vals = []
transactions = []
for index, row in df.iterrows():
    empty_set = set()
    transactions.append(empty_set)

    labels = {}
    transaction = list(row)
    if 'Cheese' in transaction:
        labels['Cheese'] = 1
        transactions[index].add('Cheese')
    else:
        labels['Cheese'] = 0
    if 'Meat' in transaction:
        labels['Meat'] = 1
        transactions[index].add('Meat')
    else:
        labels['Meat'] = 0
    if 'Bread' in transaction:
        labels['Bread'] = 1
        transactions[index].add('Bread')
    else:
        labels['Bread'] = 0
    if 'Diaper' in transaction:
        labels['Diaper'] = 1
        transactions[index].add('Diaper')
    else:
        labels['Diaper'] = 0
    if 'Milk' in transaction:
        labels['Milk'] = 1
        transactions[index].add('Milk')
    else:
        labels['Milk'] = 0
    if 'Pencil' in transaction:
        labels['Pencil'] = 1
        transactions[index].add('Pencil')
    else:
        labels['Pencil'] = 0
    if 'Bagel' in transaction:
        labels['Bagel'] = 1
        transactions[index].add('Bagel')
    else:
        labels['Bagel'] = 0
    if 'Wine' in transaction:
        labels['Wine'] = 1
        transactions[index].add('Wine')
    else:
        labels['Wine'] = 0
    if 'Eggs' in transaction:
        labels['Eggs'] = 1
        transactions[index].add('Eggs')
    else:
        labels['Eggs'] = 0

    encoded_vals.append(labels)

#adding the populated list with multiple dictionaries to a data frame
ohe_df = pd.DataFrame(encoded_vals)

#calling the apriori algorithm informing some parameters
freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

#iterate the rules data frame and print the apriori algorithm results by using the following format:

#Meat, Cheese -> Eggs
#Support: 0.21587301587301588
#Confidence: 0.6666666666666666
#Prior: 0.4380952380952381
#Gain in Confidence: 52.17391304347825
for index, row in rules.iterrows():
    supportCount = 0
    consequent = set(row[1])
    for transaction in transactions:
        if consequent.issubset(transaction):
            supportCount += 1
    prior = supportCount / len(transactions)
    print(set(row[0]), "->", set(row[1]))
    print("Support:", row[4])
    print("Confidence:", row[5])
    print("Prior:", prior)
    print("Gain in Confidence:", 100*(row[5] - prior) / prior )


#To calculate the prior and gain in confidence, find in how many transactions the consequent of the rule appears (the supporCount below). Then,
#use the gain formula provided right after.
#prior = suportCount/len(encoded_vals) -> encoded_vals is the number of transactions
#print("Gain in Confidence: " + str(100*(rule_confidence-prior)/prior))
#-->add your python code below

#Finally, plot support x confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()