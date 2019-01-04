# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 11:50:06 2019

@author: Nainesh
"""

import numpy as np

class decision_tree(object):
 
    
  def __init__(self):
    pass
 
       
  def select_question(self,rows):
    gini_max = 1.0 - 1.0/len(np.unique( rows[:,np.shape(rows)[1]-1].T ))
    gini_min = gini_max
    for i in range(0,np.shape(rows)[1]-1):
      for j in range(0,np.shape(rows)[0]):
        question = [i,rows[j,i]]                             # question have [column, value]
        gini = self.cal_gini(rows,question)
        if gini <= gini_min:
          gini_min = gini
          que = question
    return que,gini_min
  
    
  def partition(self,rows,question):
    true_rows = []
    false_rows = []
    for i in range(0,np.shape(rows)[0]):
      if (self.ans_que(question,rows[i])):
        true_rows.append(rows[i].tolist())
      else:
        false_rows.append(rows[i].tolist())
    return np.array(true_rows),np.array(false_rows)
        
        
  def ans_que(self,question,row):
    if (question[1]>='0' and question[1]<='9'):
      if row[question[0]] >= question[1]:
        return 1
    else:
      if row[question[0]] >= question[1]:
        return 1
    return 0
    
             
  def cal_gini(self,rows,question):                           # it's gonna SPLIT
    true_rows,false_rows = self.partition(rows,question)
    unique, counts = np.unique( true_rows[:,np.shape(true_rows)[1]-1].T, return_counts=True)
    gini_true = 1 -  np.sum( np.true_divide( counts**2 / np.shape(true_rows)[0] ) ) 
    unique, counts = np.unique( false_rows[:,np.shape(false_rows)[1]-1].T, return_counts=True)
    gini_false = 1 -  np.sum( np.true_divide( counts**2 / np.shape(false_rows)[0] ) ) 
    gini = np.true_divide( np.shape(true_rows)[0]*gini_true + np.shape(false_rows)*gini_false / np.shape(rows)[0] )
    return gini
        
    
  def build_tree(self,rows):
    
    question,gini = self.select_question(rows)
    if gini==0:
      Leaf = [question,rows]
      return Leaf
    
    true_rows,false_rows = self.partition(rows,question)
    
    true_branch = self.build_tree(true_rows)
    false_branch = self.build_tree(false_rows)
    
    Node = [question,true_branch,false_branch]
    return Node
    
    
    
    
    
    
    
    