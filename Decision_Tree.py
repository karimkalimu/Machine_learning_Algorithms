import requests 
import pandas as pd 
import io
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree

URL = requests.get("https://gist.githubusercontent.com/chaityacshah/899a95deaf8b1930003ae93944fd17d7/raw/3d35de839da708595a444187e9f13237b51a2cbe/pima-indians-diabetes.csv").content
data = pd.read_csv(io.StringIO(URL.decode('utf-8')))
x_train= data.drop(['9. Class variable (0 or 1)'], axis = 1)
y_train= data.get(['9. Class variable (0 or 1)'])
x = np.asarray(x_train)
y = np.asarray(y_train)
#f, ax = plt.subplots(figsize=(10, 8))
#correlation = data.corr()
#sns.heatmap(correlation, mask=np.zeros_like(correlation, dtype=np.bool), cmap=sns.diverging_palette(111, 111, as_cmap=True),
            #square=True, ax=ax)
#x_train.head()

def GiniScore( L, R, x_t, y_t):#L- Left entries, x_t the portion from x after the last split
    noise = np.random.rand(1)*0.00000000001
    # if two split have the same GiniScore we can add very small difference between them
    _ , countLeft = np.unique(y_t[L] , return_counts = True)
    _ , countRight = np.unique(y_t[R] , return_counts = True)

    if countLeft.size == 0 :
        ScoreLeft = 0
    elif countLeft.size == 1:
        ScoreLeft = 1
    else :
        ScoreLeft = (countLeft[0]/sum(countLeft))**2 + (countLeft[1]/sum(countLeft))**2

    if countRight.size == 0 :
        ScoreRight = 0
    elif countRight.size == 1:
        ScoreRight = 1
    else :
        ScoreRight = (countRight[0]/sum(countRight))**2 + (countRight[1]/sum(countRight))**2

    ScoreAll =( ScoreLeft*sum(countLeft))/(sum(countRight)+sum(countLeft)) + ( ScoreRight*sum(countRight))/(sum(countRight)+sum(countLeft))

    return ScoreAll + noise#Higher Better

def Assign_Classes( L, R, x_t, y_t):#L- Left entries,R- Right

    Classes_Left , countLeft = np.unique(y_t[L] , return_counts = True)
    Classes_Right , countRight = np.unique(y_t[R] , return_counts = True)

    if countLeft.size == 1:
        Left_Class = Classes_Left
    elif countLeft[0] > countLeft[1] :
        Left_Class = Classes_Left[0]  
    else  :
        Left_Class = Classes_Left[1]

    if countRight.size == 1:
        Right_Class = Classes_Right
    elif  countRight[0] > countRight[1] :
        Right_Class = Classes_Right[0] 
    else :
        Right_Class = Classes_Right[1]

    #print('countLeft',countLeft,Left_Class,'countRight',countRight,Right_Class)
    return Left_Class, Right_Class


def split(feature, threshold, x_t, minimum):
  
    No_Problem = False#if the both branches have the minimum
    #Less than or equal the  threshold
    L = np.where( x_t[:,feature] <= threshold )
    L = np.squeeze(np.asarray(L))
    #see the shape of L before this operation you will know why I did this :) and I like arrays more than tuple
    L_size = np.size(L)
    #More than the threshold
    R = np.where( x_t[:,feature] > threshold )
    R = np.squeeze(np.asarray(R))
    R_size = np.size(R)
    if L_size >= minimum and R_size >= minimum :
        No_Problem = True
    return  L, R, No_Problem

def SearchAndReturnF_T_G(x_t, y_t, minimum):
    #Search for the maximum giniscore in all features and values in x_t w.r.t y_t
    #minimum : minimum number in each branch
    maxGiniScore = 0
    feature = 0
    threshold = 0
    Left_Indices = 0
    Right_Indices = 0
    GiniScore_temp = 0
    for i in range(x_t.shape[1]):# Loop in  each feature
        unique_values  = np.unique(x_t[:,i]) #all possible threshold
        for j in unique_values :# Loop in each possible threshold
            L, R, No_Problem = split(i, j, x_t, minimum)
            if No_Problem :
                GiniScore_temp = GiniScore( L, R, x_t, y_t)
                if GiniScore_temp > maxGiniScore :
                    maxGiniScore = GiniScore_temp
                    feature = i
                    threshold = j
                    Left_Indices = L
                    Right_Indices = R

    return feature, threshold, maxGiniScore, Left_Indices, Right_Indices

class TreeNode():
  
    def __init__(self, parent, feature, threshold):
        self.Parent = parent
        self.Feature = feature
        self.Threshold = threshold
        self.Left_Children = 0
        self.Right_Children = 0
        self.Left_Class = 0
        self.Right_Class = 0
        
    def SetLeftChild(self, left_Children):
        self.Left_Children = left_Children
        
    def SetRightChild(self, right_Children):
        self.Right_Children = right_Children
        
    def Set_Classes(self, left_Class, right_Class):
        self.Left_Class = left_Class
        self.Right_Class = right_Class        

def recursion( x_t, y_t, minimum, Current_Depth, Max_Depth, Minimum_GiniScore, parent, left, counter):
    #left for when assigning the parent children
    #parent to know if it's root and to which node assign children
    #counter for count how many time we called this function
    #minimum :  minimum number in each branch
    #global counter
    global counter_Append
    global Nodes
    global Tree
    #print(parent, "<parent", len(Tree), "<Tree", counter)
    maxNumberNodes = 4**Max_Depth
    if counter <= maxNumberNodes :
        feature, threshold, GiniScore, Left_Indices, Right_Indices = SearchAndReturnF_T_G( x_t, y_t, minimum)  

        if GiniScore > Minimum_GiniScore :#after here we accept the value for sure
            if parent == -1 :# Root 
              #Create tree node
                Tree.append(TreeNode(parent, feature, threshold))
                #give it parent = 0, feature, threshold
                if not Current_Depth >= Max_Depth : #we already create the tree node
                  
                    if not np.size(Left_Indices) == minimum :#beacuse if it's equal we don't need to call recursion,but it's accepted
                      #print(counter)
                        counter +=1
                        Current_Depth +=1
                        left = True
                        parent = 0
                        recursion( x_t[Left_Indices], y_t[Left_Indices], minimum, Current_Depth, Max_Depth, Minimum_GiniScore, parent, left, counter)
                        Current_Depth -=1#to call the secound child with the same Depth
                    if not np.size(Right_Indices) == minimum :
                        #print(counter)
                        counter +=1
                        Current_Depth +=1
                        left = False
                        #the 0 is the parent
                        recursion( x_t[Right_Indices], y_t[Right_Indices], minimum, Current_Depth, Max_Depth, Minimum_GiniScore, 0, left, counter)
                        Current_Depth -=1#no reason until now
            else :
                #Create tree node
                #give it parent, feature, threshold, assign to parent node this children
                if left :#left or right 
                  #here actually we create the tree node if we didn't create it in parent
                    #print(counter, len(Tree),"left",parent)

                    Tree.append(TreeNode(parent, feature, threshold))
                    Tree[parent].SetLeftChild(len(Tree)-1)
                    if not Current_Depth >= Max_Depth : #we already create the tree node

                        parent = len(Tree)-1
                        if not np.size(Left_Indices) == minimum :#beacuse if it's equal we don't need to call recursion,but it's accepted
                            counter +=1
                            Current_Depth +=1
                            left = True
                            #print(counter, len(Tree),"left_l",parent)
                            recursion( x_t[Left_Indices], y_t[Left_Indices], minimum, Current_Depth, Max_Depth, Minimum_GiniScore, parent, left, counter)
                            Current_Depth -=1#to call the secound child with the same Depth
                        if not np.size(Right_Indices) == minimum :
                            counter +=1
                            Current_Depth +=1
                            left = False
                            #print(counter, len(Tree),"right_l",parent)
                            recursion( x_t[Right_Indices], y_t[Right_Indices], minimum, Current_Depth, Max_Depth, Minimum_GiniScore, parent, left, counter)
                            Current_Depth -=1#no reason until now
                else :
                  #here actually we create the tree node if we didn't create it in parent or left

                    Tree.append(TreeNode(parent, feature, threshold))
                    Tree[parent].SetRightChild(len(Tree)-1)
                    #print(counter, len(Tree),"right",parent)
                    if not Current_Depth >= Max_Depth : #we already create the tree node
                      
                        parent = len(Tree)-1
                        if not np.size(Left_Indices) == minimum :#beacuse if it's equal we don't need to call recursion,but it's accepted
                            counter +=1
                            Current_Depth +=1
                            left = True
                            #print(counter , len(Tree),"left_r",parent)
                            recursion( x_t[Left_Indices], y_t[Left_Indices], minimum, Current_Depth, Max_Depth, Minimum_GiniScore, parent, left, counter)
                            Current_Depth -=1#to call the secound child with the same Depth
                        if not np.size(Right_Indices) == minimum :
                            counter +=1
                            Current_Depth +=1
                            left = False
                            #print(counter ,len(Tree),"right_r",parent)
                            recursion( x_t[Right_Indices], y_t[Right_Indices], minimum, Current_Depth, Max_Depth, Minimum_GiniScore, parent, left, counter)
                            Current_Depth -=1#no reason until now
  #print( counter,    Current_Depth,"bye", len(Tree))

def MakeDecisionTree(x, y, minimum = 3, Max_Depth = 3, Minimum_GiniScore = 0.1):
    
    minimum = 3
    temp = 0
    counter = 0
    recursion(x, y, minimum, 0, Max_Depth, 0.1, -1, False, 0)
    #def recursion( x_t, y_t, minimum, Current_Depth, Max_Depth, counter, Minimum_GiniScore, parent, left):
    for i in range(len(Tree)):#assigningclasses
        L , R, Noproblem = split(Tree[i].Feature, Tree[i].Threshold, x, minimum)
        L_C, R_C = Assign_Classes(L, R, x, y)
        Tree[i].Set_Classes(L_C, R_C)
    return True
def predict(x_t):
    j = 0
    class_x_t = 0
    while True:
        if x_t[Tree[j].Feature] > Tree[j].Threshold :
            
            if not Tree[j].Right_Children == 0 :#Have child
                j = Tree[j].Right_Children
            else :#Don't have child so return his class
                class_x_t = Tree[j].Right_Class
                return class_x_t
        else :
            
            if not Tree[j].Left_Children == 0 :#Have child
                j = Tree[j].Left_Children
                #print("hi2",i)
            else :#Don't have child so return his class
                class_x_t = Tree[j].Left_Class#Right_Class#Left_Class#######################
                return class_x_t


#x = np.random.random((1000,2))*1000
#y = np.random.normal(12, 2, (1000,1))
if __name__ == "__main__":
    minimum = 3
    temp = 0
    counter = 0
    Minimum_GiniScore = 1e-10 
    Max_Depth = 3
    Tree = []
    MakeDecisionTree(x, y, minimum, Max_Depth, Minimum_GiniScore)
    y_predict = np.zeros(len(y))
    for i in range(len(y)):
        y_predict[i] = predict(x[i])
    clf = tree.DecisionTreeClassifier(max_depth=1 ,min_samples_split = 3)
    clf = clf.fit(x[:], y[:])
    y_sklearn = clf.predict(x)    
    myclass , mycount= np.unique(y_predict, return_counts = True)
    realclass , realcount = np.unique(y , return_counts = True)
    sklearn_class , sklearn_count = np.unique(y_sklearn , return_counts = True)  
    counter_sklearn = 0
    counter_me = 0 

    JustOnes = np.asarray(np.where( y[:] == 1 ))[0]    
    for i in range(len(y)):
        if y[i] == y_predict[i]:
            counter_me += 1
        if y[i] == y_sklearn[i]:
            counter_sklearn +=1
    print('counter_me',counter_me, 'counter_sklearn',counter_sklearn,  myclass ,
          mycount, sklearn_class , sklearn_count, 
          realclass , realcount)#not very good ,I don't know why  


        
