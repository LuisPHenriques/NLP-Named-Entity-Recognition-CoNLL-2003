import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

#---------------------------------------------------------------------
#Function to plot a graph based on the results derived
#---------------------------------------------------------------------    
def plot_loss(accuracy_measures, title):
    # Set the plot style and background color
    sns.set_style("ticks")
    
    plt.figure(figsize=(15, 8))
    
    # Get the number of experiments
    num_experiments = len(accuracy_measures)
    
    # Define the color palette
    palette = sns.color_palette("viridis", num_experiments)
    
    # Iterate over the experiments and set the line color
    for i, (experiment, accuracy) in enumerate(accuracy_measures.items()):
        color = palette[i % num_experiments]  # Select color from the palette
        plt.plot(accuracy, 
                 label=experiment,
                 linewidth=3,
                 color=color)
    
    # Remove the top and right spines of the plot
    sns.despine()
    
    plt.title(title, fontsize=18)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(loc='lower left')
    plt.grid(False)
    plt.show()


#---------------------------------------------------------------------
#Function to find the correct metric in history.history (when necessary)
#--------------------------------------------------------------------- 
def get_key(dictionary, prefix):
    for key in dictionary:
        if key.startswith(prefix):
            return key
        

#---------------------------------------------------------------------
#Function to evaluate a model for a multilabel classification problem
#--------------------------------------------------------------------- 
def evaluate_model(y_true, y_pred):
    # Function to remove padding from predictions
    def remove_padding(target_list, predictions):
        preds = []
        for i in range(len(target_list)):
            target_length = len(target_list[i])
            preds.append(predictions[i][:target_length])
        return preds
    
    y_pred = np.array(tf.nn.softmax(y_pred))

    y_true = [seq.tolist() for seq in y_true.tolist()]

    y_pred = remove_padding(y_true, y_pred)

    y_pred = [item for sublist in y_pred for item in sublist]

    predictions = []
    for i in range(len(y_pred)):
        predictions.append(np.argmax(y_pred[i]))

    y_pred = np.array(predictions)

    y_true = np.array([item for sublist in y_true for item in sublist])

    # Calculate confusion matrix
    confusion = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", confusion)
    print()
    print()
    # Calculate classification report
    report = classification_report(y_true, y_pred)
    print("Classification Report:\n", report)
        