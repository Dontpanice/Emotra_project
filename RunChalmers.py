
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.model_selection import GridSearchCV
import pickle

#with open('Dataset.pickle', 'wb') as f:
#    pickle.dump(Dataset, f)
    
with open('./Dataset.pickle', 'rb') as f:
    Dataset = pickle.load(f)



Area = Dataset[['Area','Area2','Area3','Area4','Area5','Area6','Area7']].reset_index(drop=True)
Amplitude = Dataset[['Amplitude','Amplitude2','Amplitude3','Amplitude4','Amplitude5','Amplitude6','Amplitude7']].reset_index(drop=True)
Risetime = Dataset[['Risetime','Risetime2','Risetime3','Risetime4','Risetime5','Risetime6','Risetime7']].reset_index(drop=True)
Settlingetime = Dataset[['Settlingtime','Settlingtime2','Settlingtime3','Settlingtime4','Settlingtime5','Settlingtime6','Settlingtime7']].reset_index(drop=True)
Arclength = Dataset[['Arclength1','Arclength2','Arclength3','Arclength4','Arclength5','Arclength6','Arclength7']].reset_index(drop=True)
HeartRate = Dataset[['HeartRate','HeartRate2','HeartRate3','HeartRate4','HeartRate5','HeartRate6','HeartRate7']].reset_index(drop=True)
#relative_means = Dataset['relative_mean'].reset_index(drop=True)
means = Dataset['means'].reset_index(drop=True)
N_amp = Dataset['N_amp'].reset_index(drop=True)
Label = Dataset['Label'].reset_index(drop=True)


# =============================================================================
#                      optimize for best recall score features
# =============================================================================
#%%
#AE +ST + N.SCR 
X = Area.join(N_amp).join(Settlingetime)
#ST + M
X = Settlingetime.join(means)
#AE +ST + N.SCR + AM
X = Area.join(N_amp).join(Settlingetime).join(Amplitude)
#AE +ST + N.SCR + AM
X = Arclength.join(N_amp).join(Settlingetime).join(Amplitude)
#AE +ST + N.SCR + AM
X = Arclength.join(N_amp).join(Settlingetime).join(Amplitude)

#X = Area.join(Amplitude).join(Risetime).join(Settlingetime).join(Arclength).join(HeartRate).join(means).join(N_amp)





Y = Label
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

      
    
param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001],'kernel': ['rbf']}

#classifier = SVC()

grid = GridSearchCV(SVC() ,param_grid,refit = True, verbose=2, scoring = 'recall')

#grid = GridSearchCV(SVC(kernel ='rbf') ,param_grid,refit = True, verbose=2, scoring = 'recall', cv=5,n_jobs=5)

grid.fit(X_train,y_train)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))


#https://scikit-learn.org/0.16/auto_examples/svm/plot_rbf_parameters.html

#%%










# Utility function to move the midpoint of a colormap to be around
# the values of interest.

#class MidpointNormalize(Normalize):
#
#    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
#        self.midpoint = midpoint
#        Normalize.__init__(self, vmin, vmax, clip)
#
#    def __call__(self, value, clip=None):
#        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
#        return np.ma.masked_array(np.interp(value, x, y))
#
###############################################################################
## visualization
##
## draw visualization of parameter effects
#
#
#
#
#
#
#plt.figure(figsize=(8, 6))
#xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
#for (k, (C, gamma, clf)) in enumerate(classifiers):
#    # evaluate decision function in a grid
#    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#    Z = Z.reshape(xx.shape)
#
#    # visualize decision function for these parameters
#    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
#    plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
#              size='medium')
#
#    # visualize parameter's effect on decision function
#    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
#    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r)
#    plt.xticks(())
#    plt.yticks(())
#    plt.axis('tight')
#
## plot the scores of the grid
## grid_scores_ contains parameter settings and scores
## We extract just the scores
#scores = [x[1] for x in grid.grid_scores_]
#scores = np.array(scores).reshape(len(C_range), len(gamma_range))
#
## Draw heatmap of the validation accuracy as a function of gamma and C
##
## The score are encoded as colors with the hot colormap which varies from dark
## red to bright yellow. As the most interesting scores are all located in the
## 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
## as to make it easier to visualize the small variations of score values in the
## interesting range while not brutally collapsing all the low score values to
## the same color.
#
#plt.figure(figsize=(8, 6))
#plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
#plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
#           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
#plt.xlabel('gamma')
#plt.ylabel('C')
#plt.colorbar()
#plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
#plt.yticks(np.arange(len(C_range)), C_range)
#plt.title('Validation accuracy')
#plt.show()