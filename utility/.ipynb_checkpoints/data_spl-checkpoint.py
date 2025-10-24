
from sklearn.model_selection import train_test_split

#
def data_split(X,y,rnd_st,tst_sz):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = tst_sz, 
                                                        random_state=rnd_st,
                                                        stratify=y)
    
    return X_train, X_test, y_train, y_test