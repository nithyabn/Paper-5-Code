import numpy as np
import pandas as pd,tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
rn=0

def lime_explainable_AI():
    db=1
    xx = np.load(f'saved data/fused_feat' + str(db) + '.npy')
    yy = np.load(f'saved data/fin_label' + str(db) + '.npy')
    # xx=pd.DataFrame(xx)
    # yy=pd.DataFrame(yy)



    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(xx, yy, test_size=0.2, random_state=42)

    X_train = X_train_o.reshape(X_train_o.shape[0], X_train_o.shape[1], 1)
    X_test = X_test_o.reshape(X_test_o.shape[0], X_test_o.shape[1], 1)

    num_classes = len(np.unique(y_train_o))
    y_train = to_categorical(y_train_o, num_classes=num_classes)
    y_test = to_categorical(y_test_o, num_classes=num_classes)


    loaded_model = tf.keras.models.load_model("saved data/p_model.h5")


    import lime
    from lime import lime_tabular
    explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train_o),
    feature_names=pd.DataFrame(X_train_o).columns,
    class_names=['Non stone', 'stone'],
    mode='classification'
    )

    feat_name = [f"feat{i}" for i in range(X_train.shape[1])]

    explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train_o),
    feature_names=feat_name,
    # class_names=['COVID-19', "Lung-Opacity", "Normal", "Viral Pneumonia", "Tuberculosis"],
    mode='classification'
    )
    exp = explainer.explain_instance(
        data_row=X_test_o[1],   # one sample row
        predict_fn=lambda x: loaded_model.predict(
            x.reshape(x.shape[0], x.shape[1], 1)   # reshape to (batch, timesteps, channels)
        ).astype(float),
        num_features=20
    )
    import matplotlib.pyplot as plt

    plt.rc('font',weight='bold',size=14)
    # Generate the LIME figure
    fig = exp.as_pyplot_figure()
    # Remove the title
    ax = fig.axes[0]   # get the main axis
    ax.set_title("")   # clear the title
    # Resize the figure
    fig.set_size_inches(10, 5)
    # Save the resized figure
    if rn: fig.savefig("Results/lime_explanation.png", bbox_inches="tight")
    plt.close(fig)  # optional, to avoid duplicate display


# lime_explainable_AI()


# exp = explainer.explain_instance(
# data_row=X_test[1],
# predict_fn=lambda x: loaded_model.predict(x).astype(float),
# # predict_fn=loaded_model.predict_proba,
# num_features=20
# )
# # Save the explanation as an HTML file

# exp.save_to_file('Results/lime_explanation.html')
# exp.as_pyplot_figure(figsize=(10,5)).savefig("Results/lime_explanation.png")
# exp.show_in_notebook(show_table=True)
