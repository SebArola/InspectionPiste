from vgg19_Binary import SupervisedDeepLearning

if __name__ == "__main__":

    fit = False
    
    
    if fit :
        model = SupervisedDeepLearning()        
        model.fitModel()
    else :
        model = SupervisedDeepLearning('vgg19_weights.h5')
        model.predict(3)
        