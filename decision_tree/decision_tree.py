# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 11:50:06 2019

@author: Nainesh
"""

import numpy as np



class decision_tree(object):
 
    
  def __init__(self,rows):                                  # must be a 2D one
    
    trow = []
    for x in rows[0]:
      trow.append(type(x))
    
    self.trow = trow
    self.rows = np.array(rows)
    
 
  def select_question(self,rows):
      
    gini_max = 1.0 - 1.0/len(np.unique( rows[:,np.shape(rows)[1]-1].T ))
    gini_min = gini_max
    que = [0,rows[0,0]]
    
    for i in range(0,np.shape(rows)[1]-1):
      for j in range(0,np.shape(rows)[0]):
        question = [i,rows[j,i]]                           # question have [column, value]
        gini = self.cal_gini(rows,question)
        if gini == 'ERR':
          continue;
        if gini <= gini_min:
          gini_min = gini
          que = question
    
    if gini_min == gini_max:
      gini_min = 'MAX'
    
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
     
    if (self.trow[question[0]] == int or self.trow[question[0]] == float):
      if float(row[question[0]]) >= float(question[1]):
        return 1
    else:
      if row[question[0]] == question[1]:
        return 1
    
    return 0
    
             
  def cal_gini(self,rows,question):   
                        
    true_rows,false_rows = self.partition(rows,question)
    if len(true_rows)==0 or len(false_rows)==0:
      return 'ERR'
    
    unique, counts = np.unique( true_rows[:,np.shape(true_rows)[1]-1].T, return_counts=True)
    gini_true = 1 -  np.sum( np.true_divide( counts**2 , np.shape(true_rows)[0]**2 ) )
    unique, counts = np.unique( (false_rows[:,np.shape(false_rows)[1]-1].T), return_counts=True)
    gini_false = 1.0 -  np.sum( np.true_divide( counts**2 , np.shape(false_rows)[0]**2 ))
    
    gini = np.true_divide( np.shape(true_rows)[0]*gini_true + np.shape(false_rows)[0]*gini_false , np.shape(rows)[0] )
    
    return gini
        
    
  def build_tree(self,rows=0):                    # must be a 2D one
      
    if type(rows) is int:
      rows = self.rows
    question,gini = self.select_question(rows)
    
    unique= np.unique( rows[:,np.shape(rows)[1]-1].T)
    if len(unique) is 0 or len(unique) is 1 or gini is 'MAX':
      Leaf = [ rows , unique ]
      return Leaf
    
    true_rows,false_rows = self.partition(rows,question)
    true_branch = self.build_tree(true_rows)
    false_branch = self.build_tree(false_rows)
    Node = [question,true_branch,false_branch]
    
    return Node


  def print_node(self,Node):
    
     if len(Node) is 2:
       print Node[0]
       return
     print Node[0]
     
     self.print_node(Node[1])
     self.print_node(Node[2])
     
     return


  def predict_row(self,Node,row): 
    
    
    if len(Node) is 2:
      return Node[1]
    if self.ans_que(Node[0],row) is 1:
      branch = Node[1]
    else:
      branch = Node[2]
    
    return self.predict_row( branch ,row)

    
  def predict(self,Node,rows):
    
    predicted = []
    for row in rows:
      predicted.append( self.predict_row( Node,row ).tolist() )
    
    return predicted
    
  def accuracy(self,Node,rows):
      
    predicted = self.predict(Node,rows)
    rows = np.array(rows)
    real = rows[:,np.shape(rows)[1]-1:np.shape(rows)[1]]
    predict_ = []
    
    for x in predicted:
      from random import randint
      predict_.append(x[randint(0, len(x)-1)])
    a = (predict_==real.flatten()).astype(int)
    
    return (sum(a)*100.0)/len(a)
      
     
    
    
    
    
    
    
    
    
    