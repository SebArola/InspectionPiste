from vgg19_Binary import SupervisedDeepLearning
import matplotlib.pyplot as plt

###################################
# Main.py : 
# 	Author : Sébastien Arola
#	Description : main class
###################################

##
# plotHistory :
# 	input :
#		history : the history to be plot
#	Descrtiption : plot the given history
## 
def plotHistory(history):
	 # Plot training & validation accuracy values
        plt.subplot(2, 1, 1)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        
        plt.legend(['Train', 'Test'], loc='upper left')

        # Plot training & validation loss values
        plt.subplot(2, 1, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        #Save the plot on the computer
        date = datetime.datetime.now()
        date =str(date.day)+"-"+str(date.month)+"-"+str(date.hour)+"-"+str(date.minute)    
        plt.savefig('vgg_19_fit_result_'+date+'.png')

        #Display the plot
        plt.show()
        

if __name__ == "__main__":

    fit = False
    
    
    if fit :
        model = SupervisedDeepLearning()        
        history = model.fitModel()
        plotHistory(history)
    else :
        model = SupervisedDeepLearning('vgg19_weights.h5')
        model.predict(3,"../Video/test_script.mp4")
        