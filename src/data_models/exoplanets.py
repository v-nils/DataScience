import pandas as pd
import os
import numpy as np
import tensorflow as tf
from scipy import ndimage
from tensorflow.keras.layers import Dense, Flatten
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import scienceplots
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


# Global variables
train_data = pd.read_csv('../../data/raw_data/exoTrain.csv')
test_data = pd.read_csv('../../data/raw_data/exoTest.csv')

plt.style.use(['science', 'ieee'])


class ExoplanetModel:

    def __init__(self):
        self.f_score = None
        self.accuracy = None
        self.recall = None
        self.precision = None
        self.loss = None
        self.acc = None
        self.model = None
        self.no_exoplanet = {}
        self.has_exoplanet = {}
        self.model_test = None
        self.model_train = None

    def read_test_data(self, data_path: str):
        self.model_test = pd.read_csv(data_path)

    def read_train_data(self, data_path: str):
        self.model_train = pd.read_csv(data_path)

    def preprocess_data(self, save_path: str, **kwargs) -> None:
        """
        Preprocess the data and save it to a CSV file
        :param save_path: Path to save the preprocessed data
        :return: None
        """

        plot = kwargs.get('plot', False)
        z_threshold = kwargs.get('z_threshold', 3)

        # Handle NaN values
        self.model_test.dropna(inplace=True)
        self.model_train.dropna(inplace=True)

        # Remove rows with labels other than 1 or 2
        self.model_test = self.model_test[self.model_test['LABEL'].isin([1, 2])]
        self.model_train = self.model_train[self.model_train['LABEL'].isin([1, 2])]

        # Convert labels to 0 and 1
        self.model_test['LABEL'] = self.model_test['LABEL'].apply(lambda x: 0 if x == 1 else 1)
        self.model_train['LABEL'] = self.model_train['LABEL'].apply(lambda x: 0 if x == 1 else 1)

        if plot:
            fig, ax = plt.subplots()
            ax.pie(self.model_train['LABEL'].value_counts(), labels=['No exoplanet', 'Exoplanet'], autopct='%1.1f%%')
            plt.show()

        # Remove outliers
        print('Removing outliers')
        print('Rows before removing outliers:', self.model_train.shape[0], '(train) and ', self.model_test.shape[0],
              '(test)')
        zscore_train = np.abs(
            (self.model_train.iloc[:, 1:] - self.model_train.iloc[:, 1:].mean()) / self.model_train.iloc[:, 1:].std())
        zscore_test = np.abs(
            (self.model_test.iloc[:, 1:] - self.model_test.iloc[:, 1:].mean()) / self.model_test.iloc[:, 1:].std())

        self.model_train = self.model_train[(zscore_train < z_threshold).all(axis=1)]
        self.model_test = self.model_test[(zscore_test < z_threshold).all(axis=1)]

        print('Rows after removing outliers:', self.model_train.shape[0], '(train) and ', self.model_test.shape[0],
              '(test)')

        # Get the label column
        labels_test = self.model_test['LABEL'].values
        labels_train = self.model_train['LABEL'].values

        # Standardize the data
        standard_scaler = StandardScaler()  # Select the scaling method

        # Standardize training and test data
        # Only standardize the features, not the labels
        self.model_test = standard_scaler.fit_transform(self.model_test.iloc[:, 1:])
        self.model_test = pd.DataFrame(self.model_test, columns=test_data.columns[1:])

        self.model_train = standard_scaler.fit_transform(self.model_train.iloc[:, 1:])
        self.model_train = pd.DataFrame(self.model_train, columns=train_data.columns[1:])

        print(self.model_train.head())

        # Apply PCA
        pca = PCA()
        pca.fit(self.model_train)

        # Print explained variance ratios for each component
        explained_variances = pca.explained_variance_ratio_
        print("Explained variance ratios for each component:", explained_variances)

        # Calculate cumulative variance
        cumulative_variance = np.cumsum(explained_variances)
        print("Cumulative variance explained by components:", cumulative_variance)

        # Select components with cumulative variance up to 90
        n_components = np.argmax(cumulative_variance >= 0.90) + 1
        print(f"Number of components to explain 90% of the variance: {n_components}")

        if plot:
            fig, ax = plt.subplots()
            ax.plot(cumulative_variance)
            ax.set_xlabel('Number of components')
            ax.set_ylabel('Cumulative variance')
            ax.axvline(n_components, color='red', linestyle='--')
            ax.set_xscale('log')
            plt.show()

        # Fit PCA with the selected number of components
        pca = PCA(n_components=n_components)

        self.model_train = pca.fit_transform(self.model_train)
        self.model_test = pca.transform(self.model_test)

        print("N-Components:", n_components)

        # Add a gaussian filter to the data
        self.model_train = ndimage.gaussian_filter(self.model_train, sigma=10)
        self.model_test = ndimage.gaussian_filter(self.model_test, sigma=10)

        # Add the labels back to the data
        self.model_test = pd.DataFrame(self.model_test)
        self.model_test['label'] = labels_test
        self.model_train = pd.DataFrame(self.model_train)
        self.model_train['label'] = labels_train

        # Apply SMOTE to handle class imbalance
        smote = SMOTE()
        self.model_train, self.model_train['label'] = smote.fit_resample(self.model_train.iloc[:, :-1],
                                                                         self.model_train['label'])

        # Save the data
        self.model_test.to_csv(os.path.join(save_path, 'exoTest_std_pca.csv'), index=False)
        self.model_train.to_csv(os.path.join(save_path, 'exoTrain_std_pca.csv'), index=False)

    def create_model(self, units: list | None = None, activation: list | None = None) -> None:

        if activation is None:
            activation = ['relu', 'relu']
        if units is None:
            units = [50, 1]

        model = tf.keras.models.Sequential([
            Flatten(),
            Dense(units[0], activation=activation[0], name='hidden_layer'),
            Dense(units[1], activation=activation[1], name='output_layer', kernel_initializer='he_normal'),
            Dense(1, activation='sigmoid')
        ])

        self.model = model

    def train_model(self, optimizer: str = 'adam', learning_rate: float = 0.05, epochs=50) -> None:

        if optimizer == 'adam':
            optimizer_instance = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            optimizer_instance = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            optimizer_instance = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            raise ValueError('Invalid optimizer')

        self.model.compile(optimizer=optimizer_instance,
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        summary = self.model.fit(self.model_train.iloc[:, :-1],
                       self.model_train['label'],
                       epochs=epochs,
                       batch_size=64,
                       shuffle=True,
                       validation_split=0.3)

        # Print the model summary
        #self.model.summary()
        self.loss = summary.history['loss']

    def evaluate_model(self) -> None:
        y_true = self.model_test['label']
        y_pred = (self.model.predict(self.model_test.iloc[:, :-1]) > 0.5).astype("int32")

        self.precision = precision_score(y_true, y_pred)
        self.recall = recall_score(y_true, y_pred)
        self.accuracy = accuracy_score(y_true, y_pred)
        self.f_score = f1_score(y_true, y_pred)

    def compute_stats(self, which: str = 'train'):

        if which == 'train':
            data_model = self.model_train
        else:
            data_model = self.model_test

        self.has_exoplanet[which] = data_model[data_model['label'] == 2]
        self.no_exoplanet[which] = data_model[data_model['label'] == 1]

    def plot_confusion_matrix(self) -> None:
        """
        Plot the confusion matrix using the test data
        :return: None
        """
        y_true = self.model_test['label']
        y_pred = (self.model.predict(self.model_test.iloc[:, :-1]) > 0).astype("int32")

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.show()

def plot_fluxes(data: pd.DataFrame):

    avg_data = data.groupby('LABEL').mean()

    fig, ax = plt.subplots(2, 1)
    ax[0].scatter(np.linspace(0, len(avg_data.loc[1])), avg_data.loc[1])
    ax[0].set_title('No Exoplanet')
    ax[1].scatter(avg_data.loc[2])
    ax[1].set_title('Exoplanet')
    plt.show()




if __name__ == '__main__':

    data = pd.read_csv('../../data/raw_data/exoTrain.csv')
    plot_fluxes(data)

    """
    # Set to True to re-preprocess the data
    pre_process = False

    path_test = '../../data/processed_data/exoTest_std_pca.csv'
    path_train = '../../data/processed_data/exoTrain_std_pca.csv'

    Exo = ExoplanetModel()

    # Check if the preprocessed data already exists. If not, preprocess and save the data
    if not os.path.exists(path_train) and not os.path.exists(path_test) or pre_process:

        path_test = '../../data/raw_data/exoTest.csv'
        path_train = '../../data/raw_data/exoTrain.csv'
        save_path = '../../data/processed_data/'

        Exo.read_test_data(path_test)
        Exo.read_train_data(path_train)
        Exo.preprocess_data(save_path=save_path, plot=True)

    else:
        # Data already preprocessed and saved: load the data

        Exo.read_train_data('../../data/processed_data/exoTrain_std_pca.csv')
        Exo.read_test_data('../../data/processed_data/exoTest_std_pca.csv')

    Exo.plot_correlations()
    

    params = {
        'units': [[i, 1] for i in range(1, 50, 1)],
        'activation': [['relu', 'sigmoid'], ['relu', 'relu'], ['tanh', 'sigmoid'], ['tanh', 'tanh']]
    }

    Exo.create_model(units=[256, 128], activation=['relu', 'relu'])
    Exo.train_model(epochs=25)
    Exo.evaluate_model()
    #Exo.plot_confusion_matrix()

    fig, ax = plt.subplots()
    ax.plot(Exo.loss)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.show()

    print("F1 Score:", Exo.f_score)"""



