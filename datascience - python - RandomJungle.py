

# ## <center><font color = 'green'> RandomJungle </font></center>

# ### Librairies Python



get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np 
import pandas as pd
import sklearn as skl


# ### Jeux de données artificiel
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html



from sklearn.datasets import make_classification

X, y = make_classification(n_samples = 60000 , 
                           n_features = 25 , 
                           n_informative = 15 ,
                           n_redundant = 0 , 
                           n_repeated = 0 , 
                           n_classes = 2 ,
                           n_clusters_per_class = 1 ,
                           weights =[ 0.99 , 0.01 ] ,
                           class_sep = 1.0 , 
                           random_state = 0 )


# ### Transformations en dataframe


# Variables explicatives :

X_var = [ 'x_' + str( k ) for k in range( 0 , X.shape[1] ) ]

# Variable à expliquer :

y_var = [ 'y' ]

# Dataframes : 

df_X = pd.DataFrame( X , columns = X_var )

df_y = pd.DataFrame( y , columns = y_var )


# ### Empilement de df_X et df_y


df = pd.concat( [ df_X , df_y ] , axis = 1 )


# ### Librairies pour modélisation


from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

import pickle


# ### Classificateurs spécialisés :

def RandomJungle( N , K ) :
    
    """
     BaseLine vs RandomForest : 
      * N : nombre d'expériences 
      * K : nombre d'échantillons de cas négatifs mis en face des cas positifs
      
      Expérience : K tirages dans les cas négatifs mis en face des cas positifs
    """
    
    lst_bsl_auc = []
    lst_randjgl_auc = []
    
    for i in range( N ) :
        
        # [ 0 ] Train / test split :
        df_train_all , df_test = train_test_split( df , train_size = 0.9 )

        df_train_0 = df_train_all[ df_train_all[ 'y' ] == 0 ]
        df_train_1 = df_train_all[ df_train_all[ 'y' ] == 1 ]
        
        # [ 0.1 ] train dataset :

        df_X_train_all = df_train_all[ X_var ]
        df_y_train_all = df_train_all[ y_var ]

        X_train_all = df_X_train_all.values
        y_train_all = np.ravel( df_y_train_all.values )

        # [ 0.2 ] test dataset :

        df_X_test = df_test[ X_var ]
        df_y_test = df_test[ y_var ]

        X_test = df_X_test.values
        y_test = np.ravel( df_y_test.values )
        
        # [ 1 ] Baseline :
        
        clf = RandomForestClassifier( n_estimators = 10 )
        #clf = svm.LinearSVC()
        #clf = LogisticRegression( random_state=0, solver='lbfgs', multi_class='ovr')
        #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2) ) 
        #clf =  AdaBoostClassifier()
        #clf = QuadraticDiscriminantAnalysis()
        #clf = GaussianNB()
        
        clf.fit( X_train_all , y_train_all )
        y_test_pred = clf.predict( X_test )
        auc = roc_auc_score( y_test , y_test_pred )
        lst_bsl_auc.append( auc )
        print( '[ i : {0} | baseline | auc ( test dataset ) : {1:.4f} ]'.format( i , auc ) )
        
        
        # [ 2 ] RandomJungle :
        
        # Dictionnaire pour stockage des K prédictions
        #pred_train = {}
        
        lst_auc = []
        # RandomJungle - Taille des échantillons à mettre en face des fraudes :
        spl_siz = len( df_train_1.index )
        
        # RandomJungle - K RandomForest()
        for j in range( 0 , K ) :
            # RandomJungle - train dataset - échantillon de sinistres non fraude (autant que de sinistres fraude dans le train dataset)
            df_train_0_spl = df_train_0.sample( n =  spl_siz ) 
            # RandomJungle - train dataset - concaténation des sinistres fraude et de l'échantillon de sinistres non fraude
            df_tmp = pd.concat( [ df_train_1 , df_train_0_spl ] , axis = 0 )
    
            df_X_train = df_tmp[ X_var ]
            df_y_train = df_tmp[ y_var ]
    
            X_train = df_X_train.values
            y_train = np.ravel( df_y_train.values )
    
            clf = RandomForestClassifier( n_estimators = 10 )
            #clf = svm.LinearSVC()
            #clf = LogisticRegression( solver='lbfgs', multi_class='ovr' )
            #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2) ) 
            #clf =  AdaBoostClassifier()
            #clf = QuadraticDiscriminantAnalysis()
            #clf = GaussianNB()
            
            clf.fit( X_train , y_train )
     
            # RandomJungle - Sauvegarde des modèles sur disque
            filename = './mdl/_mdl_' + str( i ) + '_' + str( j ) + '.sav'
            pickle.dump( clf , open( filename , 'wb' ) )
    
            # RandomJungle - Prédictions sur test dataset
            y_test_pred = clf.predict( X_test )
    
            # RandomJungle - AUC sur test dataset
            auc = roc_auc_score( y_test , y_test_pred )
            lst_auc.append( auc )
            #print( 'i : {0} | j : {1} | RandomForest | auc (test dataset) : \n {2}'.format( i , j , auc ) )
    
        
        pred = {}

        for j in range( K ) :
            filename = './mdl/_mdl_' + str( i ) + '_' + str( j ) + '.sav'
            clf = pickle.load( open( filename , 'rb' ) )
            y_test_pred = clf.predict( X_test )
            col = 'y_pred_' + str( j )
            pred[ col ] = y_test_pred
    
        df_X_test_pred = pd.DataFrame( pred )
        df_X_test_pred = df_X_test_pred[ sorted( df_X_test_pred.columns ) ]
        
        df_X_test_pred[ 'cons' ] = df_X_test_pred.sum( axis = 1 ).apply( lambda x : ( K - x ) / K )
        df_X_test_pred[ 'y_pred' ] = np.where( df_X_test_pred[ 'cons' ] < 0.50 , 1 , 0 )
        df_X_test_pred[ 'y_real' ] = y_test
        
        auc = roc_auc_score( y_test , df_X_test_pred[ 'y_pred' ] )
        print( '[ i : {0} | RandomJungle | auc ( test dataset ) : {1:.4f} ]'.format( i , auc ) )
        lst_randjgl_auc.append( auc )
 
    return lst_bsl_auc , lst_randjgl_auc 


# ### N Expériences ( pour chaque expérience K RandomForest = RandomJungle :

bsl_vs_randjgl_auc = RandomJungle( N = 100 , K = 500 )


# ### <font color = 'orange'>Différence significative entre AUC baseline vs RandomJungle ?</font>



np.mean( bsl_vs_randjgl_auc[1] )



np.std( bsl_vs_randjgl_auc[1] )




from scipy.stats import ttest_ind

ttest_ind( bsl_vs_randjgl_auc[0] , bsl_vs_randjgl_auc[1] )




import matplotlib.pyplot as plt

h = [ np.mean( bsl_vs_randjgl_auc[0] ) , np.mean( bsl_vs_randjgl_auc[1] ) ]
bar_lbl = [ 'AUC(Baseline)' , 'AUC (RandomJungle)' ]
y_pos = np.arange( len( bar_lbl ) )

plt.barh( y_pos , h )

plt.yticks( y_pos , bar_lbl )

plt.show()

