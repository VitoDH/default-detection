Default Detection on P2P lending

#### **Author**: Dehai Liu

**Department of Mathematics, Sun Yat-Sen University**



## 1. Abstract

This project mainly focus on data mining in the P2P lending data. Based on LDA topic model, we can extract information from the loan statement given by the borrower. Combined with the traditional features (gender,age,target value,etc) in default detection, we are able to predict the probability of default with random forest, which demonstrates high accuracy and straightforward interpretation.



## 2. Data Description

`Ren Ren Dai`(https://www.renrendai.com/) is one of the largest online P2P lending platform in China.  Using the crawling software Octopus, we could obtain around 10,000 records of online lending with the 20 features and 1 label (default or not default) :



**Lending Hard Information**

|   Attributes   |          Definitions           |
| :------------: | :----------------------------: |
| interest rate  | the interest rate of the loan  |
|  target value  |     the amount of the loan     |
| lending period | the period of holding the loan |



**Lending Soft Information**

|    Attributes    |                         Definitions                          |
| :--------------: | :----------------------------------------------------------: |
| repayment method | 1: pay interest first 2: pay interest and principal together |
|  property loan   |     1: short-term turnover 2: personal consumption ....      |
|  loan statement  |          the statement for loan before application           |



**Lending Personal Info**

|   Attributes    |                    Definitions                    |
| :-------------: | :-----------------------------------------------: |
|       age       |                continuous variable                |
|     gender      |            0: female, 1: male, 2: NULL            |
|    education    | 1: high school 2: junior college 3: undergrad ... |
|    marriage     |     1: divorce 2: married 3: single 4: widow      |
| census register |                 province in China                 |
|     income      |             1: <1000 2: 1000-2000 ...             |
| house property  |                   1: yes 0: no                    |
|   house loan    |                   1: yes 0: no                    |
|       car       |                   1: yes 0: no                    |
|    car_loan     |                   1: yes 0: no                    |
|  type company   |                  type of company                  |
|    industry     |                 1: IT 2: food ...                 |
|  scale company  |        (# of staffs) 1: <10 2: 10-100 ...         |
|    type job     |                    type of job                    |
|    workplace    |                the location of job                |
|    time job     |            1: <1 year 2: 1-3 years ...            |



## 3. Data Preprocessing

### (a) Load the data

* Load p2pData.csv

Note : the irrelevant feature in the raw data has been removed



### (b) Outlier Detection

We only perform outlier detection on continuous variables **target value** and **age** with Tukey Mehod, i.e defining the data points out of 1.5 times the Interquartile range as outleir.



**Target Value**

*  Before removing the outlier

<img src="https://github.com/VitoDH/default-detection/raw/master/img/with_outlier_Target.png" style="zoom:60%" />

* After removing the outlier

<img src="https://github.com/VitoDH/default-detection/raw/master/img/without_outlier_Target.png" style="zoom:60%" />



**Age**

* Before removing the outlier

<img src="https://github.com/VitoDH/default-detection/raw/master/img/with_outlier_Age.png" style="zoom:60%" />

* After removing the outlier

<img src="https://github.com/VitoDH/default-detection/raw/master/img/without_outlier_Age.png" style="zoom:60%" />





## 4. Feature Engineering

### a. Direct Link

Given a specific buyer and seller, it's not difficult to find out the times of the four actions that the buyers have done in the store of the merchants. Here we define a 6-dimension vector to denote the direct link between the user and the merchant:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;v_d=(gender,age,click,cart,buy,favorite)" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />

### b. Indirect Link

##### (1) First we define the weight for different action types:

|     Action      | Value | Times | Weight |
| :-------------: | :---: | :---: | :----: |
|      Click      |   0   |  n_1  |  0.1   |
|   Add to Cart   |   1   |  n_2  |  0.2   |
|       Buy       |   2   |  n_3  |  0.3   |
| Add to Favorite |   3   |  n_4  |  0.4   |



##### (2) Vitality and Popularity

Vitality is used to measure the extent of how much the user loves shopping. Popularity refers to how attractive the merchant's commodity is. They are both calculated by the weighted average of the four actions. And they can be specifically defined as **category vitality**, **brand vitality** for the user and **category popularity**, **brand popularity** for the merchant.



Now we illustrate the calculation by taking the **category vitality** as an example.



The score of a good<img src="https://latex.codecogs.com/svg.latex?\Large&space;j" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /> for a given user <img src="https://latex.codecogs.com/svg.latex?\Large&space;i" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /> is



<img src="https://latex.codecogs.com/svg.latex?\Large&space;score_{ij}=0.1*n_1+0.2*n_2+0.3*n_3+0.4*n_4" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />



The  **category vitality** of user <img src="https://latex.codecogs.com/svg.latex?\Large&space;i" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /> is



<img src="https://latex.codecogs.com/svg.latex?\Large&space;S_{cat-vitality}^i=mean(score_{ij})" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />



where <img src="https://latex.codecogs.com/svg.latex?\Large&space;n_i" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /> refers the item that is relevant to the user <img src="https://latex.codecogs.com/svg.latex?\Large&space;i" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />.



Similarly, we can calculate the other three indicators and combine them into a 4-dimension vector.

  











### c. Normalization

After setting up the features, we find out the each feature have different scales. Thus, it would be reasonable to scale the attributes to [0,1] using the following formula:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{x_i-min(x_i)}{max(x_i)-min(x_i)}" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />



### d. PCA

Based on part a and b, we have obtained **10** features. In order to simplify the training process and remove useless information, we perform PCA on the training set. The scree plot and the variance of components are given as follows:

 <img src="https://github.com/VitoDH/repeated_buyers/raw/master/img/scree_plot.png" style="zoom:90%" />

<img src="https://github.com/VitoDH/repeated_buyers/raw/master/img/pca_var.png" style="zoom:90%" />



Noting that the cumulative proportion has reach **0.94** on the 4th principal component, we can simply pick the first four principal components as our attributes for training.





## 5. Balance 

Taking a glance at the distribution of the label, 

|        Type        | Number |
| :----------------: | :----: |
|    Total Sample    | 90917  |
| Positive Label (1) |  5363  |
| Negative Label (0) | 85554  |
|   Number of User   | 75053  |
| Number of Merchant |  1982  |



From the above table, the positive samples only covers 5.9% of the total samples, which will easily lead to a situation that all the positive label will be classified as negative.

We use four ways of sampling to address this problem and obtain a balance dataset.



|     Label      |   0   |   1   |
| :------------: | :---: | :---: |
|      Raw       | 84700 | 5300  |
| Over Sampling  | 84700 | 84700 |
| Under Sampling | 5300  | 5300  |
| Over and Under | 44959 | 45041 |
|     SMOTE      | 44959 | 45041 |





## 6. Training with XG Boost

XG Boost is an cutting-edge algorithm derived from GBDT , which can deal with missing data and avoid overfitting.

### a. Parametrization

|    Parameters     |  Value   |
| :---------------: | :------: |
|     max_depth     |    5     |
|   learning_rate   |   0.1    |
|     max_iter      |   800    |
| learning_function | logistic |



### b. Performance under different sampling

#### (1) Over Sampling

<img src="https://github.com/VitoDH/repeated_buyers/raw/master/img/xg_over_sample.png" style="zoom:90%" />

|               | **Train** | **Test** |
| :-----------: | :-------: | :------: |
| **Precision** |   0.783   |  0.128   |
|  **Recall**   |   0.846   |  0.444   |
| **F1 Score**  |   0.814   |  0.200   |
| **F2 Score**  |   0.832   |  0.297   |
|    **AUC**    |   0.885   |  0.603   |



#### (2) Under Sampling

<img src="https://github.com/VitoDH/repeated_buyers/raw/master/img/xg_under_sample.png" style="zoom:90%" />

|               | **Train** | **Test** |
| :-----------: | :-------: | :------: |
| **Precision** |   0.833   |   0.08   |
|  **Recall**   |   0.862   |  0.667   |
| **F1 Score**  |   0.848   |  0.144   |
| **F2 Score**  |   0.856   |  0.270   |
|    **AUC**    |   0.929   |  0.549   |



#### (c) Both

<img src="https://github.com/VitoDH/repeated_buyers/raw/master/img/xg_both.png" style="zoom:90%" />

|               | **Train** | **Test** |
| :-----------: | :-------: | :------: |
| **Precision** |   0.817   |  0.119   |
|  **Recall**   |   0.846   |  0.460   |
| **F1 Score**  |   0.832   |  0.190   |
| **F2 Score**  |   0.839   |  0.293   |
|    **AUC**    |   0.909   |  0.609   |



#### (d) SMOTE

<img src="https://github.com/VitoDH/repeated_buyers/raw/master/img/xg_smote.png" style="zoom:90%" />

|               | **Train** | **Test** |
| :-----------: | :-------: | :------: |
| **Precision** |   0.727   |  0.143   |
|  **Recall**   |   0.687   |  0.460   |
| **F1 Score**  |   0.706   |  0.218   |
| **F2 Score**  |   0.695   |  0.318   |
|    **AUC**    |   0.789   |  0.641   |



## 7. Conclusion

From the results above, we are able to conclude that:

* Model successfully captures the information in the dataset, represented by high F1 score and AUC in the training set.
* Model can be used to detect whether a buyer will come back again to a specific online store as long as the data between them is given.
* For improvement in the test set, we need to focus more on the feature engineering part and the sampling part.

