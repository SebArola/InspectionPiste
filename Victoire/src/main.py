from vgg19_Binary import VGG_19_Binary
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import confusion_matrix
import numpy as np



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
def plotHistory(history, ficName):
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
		plt.savefig(ficName+date+'.png')

		#Display the plot
		plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax

if __name__ == "__main__":

	fit = True
	
	
	if fit :
		model = VGG_19_Binary()		
		history = model.fitModel(16,10)
		plotHistory(history[0],'vgg_19_fit_result_')
		cm_plot_labels = ["no_debris","debris"]
		plot_confusion_matrix(history[1], cm_plot_labels,normalize=True,title="Confusion Matrix")
	else :
		model = VGG_19_Binary('vgg19_weights.h5')
		model.predict(3,"../Video/test_script.mp4")
		