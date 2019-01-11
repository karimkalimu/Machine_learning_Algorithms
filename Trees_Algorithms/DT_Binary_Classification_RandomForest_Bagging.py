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
    
  print('countLeft',countLeft,Left_Class,'countRight',countRight,Right_Class)
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

def MakeDecisionTree(x, y, minimum = 3, Max_Depth = 3, Minimum_GiniScore = 0.1):
    global Tree
    minimum = 3
    counter = 0
    recursion(x, y, minimum, 0, Max_Depth, 0.1, -1, False, 0)
    #def recursion( x_t, y_t, minimum, Current_Depth, Max_Depth, counter, Minimum_GiniScore, parent, left):
    for i in range(len(Tree)):
        L , R, Noproblem = split(Tree[i].Feature, Tree[i].Threshold, x, minimum)
        L_C, R_C = Assign_Classes(L, R, x, y)
        Tree[i].Set_Classes(L_C, R_C)
    return Tree

def downsampling(x_t, y_t, Ratio):
    Classes , counts = np.unique(y_t, return_counts = True)
    Class_1_Indices = np.asarray(np.where( y_t[:] == Classes[1] ))[0]
    Class_0_Indices = np.asarray(np.where( y_t[:] == Classes[0] ))[0]
    Difference = abs(int((counts[0] - counts[1])*(1-((Ratio + 1e-14)/100))))
    #when the ratio 100 after downsampling the two classes will have the same size
    if counts[0] > counts[1] :#if class_0 have more points than class_1
        counts_temp = counts[1] + Difference
        Indices_temp = np.random.choice(Class_0_Indices,counts_temp, replace= True)
        y_t = np.concatenate((y_t[Class_1_Indices], y_t[Indices_temp]))
        x_t = np.concatenate((x_t[Class_1_Indices], x_t[Indices_temp]))
        return x_t, y_t
    elif counts[1] > counts[0] :
        counts_temp = counts[0] + Difference
        Indices_temp = np.random.choice(Class_1_Indices,counts_temp, replace= True)
        y_t = np.concatenate((y_t[Class_0_Indices], y_t[Indices_temp]))
        x_t = np.concatenate((x_t[Class_0_Indices], x_t[Indices_temp]))
        return x_t, y_t
    else :#if equal
        return x_t, y_t
def upsampling(x_t, y_t, Ratio):
    Classes , counts = np.unique(y_t, return_counts = True)
    Class_1_Indices = np.asarray(np.where( y_t[:] == Classes[1] ))[0]
    Class_0_Indices = np.asarray(np.where( y_t[:] == Classes[0] ))[0]
    Difference = abs(int((counts[0] - counts[1])*(((Ratio + 1e-14)/100))))
    #when the ratio 100 after upsampling the two classes will have the same size
    if counts[0] > counts[1] :#if class_0 have more points than class_1
        counts_temp = counts[1] + Difference
        Indices_temp = np.random.choice(Class_1_Indices,counts_temp, replace= True)
        y_t = np.concatenate((y_t[Class_0_Indices], y_t[Indices_temp]))
        x_t = np.concatenate((x_t[Class_0_Indices], x_t[Indices_temp]))
        return x_t, y_t
    elif counts[1] > counts[0] :
        counts_temp = counts[0] + Difference
        Indices_temp = np.random.choice(Class_0_Indices,counts_temp, replace= True)
        y_t = np.concatenate((y_t[Class_1_Indices], y_t[Indices_temp]))
        x_t = np.concatenate((x_t[Class_1_Indices], x_t[Indices_temp]))
        return x_t, y_t
    else :#if equal
        return x_t, y_t  
def Boot_Strapping(x_t, y_t, Ratio):
    #Ratio here some how what is the size of the sample you want
    #Example : if we have 1000 point ,ratio 30 ,we will choiche randomly from all point (30/100)*1000 and return them
    #We are not choicing the point w.r.t classes 
    data = np.concatenate((x_t, y_t), axis = 1)
    Sample_size = int(len(data)*(Ratio/100 ))
    point_Indices = np.arange(len(data))
    point_Indices = np.random.choice(point_Indices,Sample_size, replace= True)
    data = data[point_Indices]
    features_len = len(x_t[0,:])
    x_t = data[:,:features_len]
    y_t = data[:,features_len:]
    return x_t, y_t
 
      
#x_t, y_t = downsampling(x, y, 10)
#x_t, y_t = upsampling(x, y, 10)


def predict_class_for_Bagging(x_t, Tree_s):
    class_x_t = np.zeros(len(Tree_s))
    for i in range(len(Tree_s)):
        Tree = Tree_s[i]
        j = 0
        while True:
            if x_t[Tree[j].Feature] > Tree[j].Threshold :

                if not Tree[j].Right_Children == 0 :#Have child
                    j = Tree[j].Right_Children
                else :#Don't have child so return his class
                    class_x_t[i] = Tree[j].Right_Class
                    break
            else :

                if not Tree[j].Left_Children == 0 :#Have child
                    j = Tree[j].Left_Children
                    #print("hi2",i)
                else :#Don't have child so return his class
                    class_x_t[i] = Tree[j].Left_Class#Right_Class#Left_Class#######################
                    break
    return   1 if np.mean(class_x_t) >= 0.5 else 0

  
def predict_class_Random_Forest(x_t, Tree_s, Features):#Features arrays of the features in each tree
    class_x_t = np.zeros(len(Tree_s))
    for i in range(len(Tree_s)):
        Tree = Tree_s[i]
        j = 0
        while True:
            if x_t[Features[i,Tree[j].Feature]] > Tree[j].Threshold :

                if not Tree[j].Right_Children == 0 :#Have child
                    j = Tree[j].Right_Children
                else :#Don't have child so return his class
                    class_x_t[i] = Tree[j].Right_Class
                    break
            else :

                if not Tree[j].Left_Children == 0 :#Have child
                    j = Tree[j].Left_Children
                    #print("hi2",i)
                else :#Don't have child so return his class
                    class_x_t[i] = Tree[j].Left_Class#Right_Class#Left_Class#######################
                    break
    return   1 if np.mean(class_x_t) >= 0.5 else 0


  
def predict(x_t, Tree):#predict class for one point
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

def Bagging(x_t, y_t, minimum, Max_Depth, Minimum_GiniScore, Number_DecisionTree = 2,
            Ratio_subsample = 100, downsampling_Raio = 0, upsampling_ratio = 0):
    global Tree
    global Tree_s
    for i in range(Number_DecisionTree):
        Tree = []
        x_t_2, y_t_2  = downsampling(x_t, y_t, downsampling_Raio)
        x_t_2, y_t_2  = upsampling(x_t_2, y_t_2, upsampling_ratio)
        x_t_2, y_t_2 = Boot_Strapping(x_t_2, y_t_2, Ratio_subsample)
        MakeDecisionTree(x_t_2, y_t_2, minimum, Max_Depth, Minimum_GiniScore)
        Tree_s.append(Tree.copy())


def Random_Forest(x_t, y_t, minimum, Max_Depth, Minimum_GiniScore, Number_DecisionTree = 2,
                  Ratio_subsample = 20, subfeature = 0, downsampling_Raio = 0):
    #subfeature : maximum number of feature to take in each tree
    global Tree
    global Tree_s
    global Features
    feauter_len =  len(x_t[0,:])
    if subfeature == 0 :#default 
        subfeature = int(np.sqrt(feauter_len))
    for i in range(Number_DecisionTree):
        Tree = []
        x_t_2, y_t_2  = downsampling(x_t, y_t, downsampling_Raio)
        feauters = np.random.choice(np.arange(feauter_len),subfeature, replace=False)
        Features[i] = feauters
        x_t_2, y_t_2 = Boot_Strapping(x_t_2[:,feauters], y_t_2, Ratio_subsample)
        MakeDecisionTree(x_t_2, y_t_2, minimum, Max_Depth, Minimum_GiniScore)
        Tree_s.append(Tree.copy())


if __name__ == "__main__":

    #x = np.random.random((1000,2))*1000
    #y = np.random.normal(12, 2, (1000,1))


    x_train= data.drop(['9. Class variable (0 or 1)'], axis = 1)
    y_train= data.get(['9. Class variable (0 or 1)'])

    minimum = 4
    Minimum_GiniScore = 1e-10 
    Max_Depth = 5
    Tree_s = []
    Tree = []
    Number_DecisionTree = 20
    Ratio_subsample = 40
    #if class_0 50 class_1 150 , difference between them = 100 , 
    #ratio 30 downsample the class with more point by 30 percent of the difference
    downsampling_Raio = 60
    #downsampling_Raio and upsampling_ratio : the Higher  the Ratio will make the two sample size are close to each other
    #but the differnece downsampling_Raio make the size of the bigger smaller , upsampling_ratioHigherthe opposit
    upsampling_ratio = 0
    subfeature = 4#default
    Features = np.zeros((Number_DecisionTree,subfeature),dtype = int)

    #MakeDecisionTree(x, y, minimum, Max_Depth, Minimum_GiniScore) # Make just one Decision Tree
    #Bagging(x_t, y_t, minimum, Max_Depth, Minimum_GiniScore, Number_DecisionTree ,Ratio_subsample, downsampling_Raio, upsampling_ratio):
    Random_Forest(x, y, minimum, Max_Depth, Minimum_GiniScore, Number_DecisionTree , Ratio_subsample, subfeature, downsampling_Raio)
    print(len(Tree_s))
    counter_me = 0
    predicted_calss = np.zeros(len(x[:,0]))
    for i in range(len(x[:,0])):#All point
        #predicted_calss[i] = predict_class_for_Bagging(x[i], Tree_s)
        predicted_calss[i] = predict_class_Random_Forest(x[i], Tree_s, Features)
    for i in range(len(y)):
        if y[i] == predicted_calss[i]:
          counter_me += 1

    myclass , mycount= np.unique(predicted_calss, return_counts = True)
    realclass , realcount = np.unique(y , return_counts = True)

    print(counter_me,  myclass , mycount)
    print(realclass , realcount)



    
    
