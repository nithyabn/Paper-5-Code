
######  import all package #########



USE_CUDA = False
import torch.optim as optim
rn=0
from other.root import *
from other.Confusion_matrix import *
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers,models
import networkx as nx
import numpy as np,math,os
import pandas as pd
import torch,time,tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if USE_CUDA else 'cpu')
from tensorflow.keras import layers, models, optimizers

########## function ##########


def dcnn_(X_train, Y_train, X_test, Y_test, db):
    # number of classes
    ln = len(set(Y_train.flatten()))
    # reshape for 1D CNN: (samples, timesteps, features)
    cnn_X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    cnn_X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # build 1D CNN
    model = models.Sequential()
    model.add(layers.Conv1D(16, kernel_size=3, padding='valid', input_shape=(cnn_X_train.shape[1], 1), activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv1D(32, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(ln, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('\n1D CNN model')
    model.summary()
    # train
    model.fit(cnn_X_train, Y_train, epochs=ep, batch_size=128, verbose=0)
    # predict
    pred = model.predict(cnn_X_test)
    y_pred = array(np.argmax(pred, axis=1),axis=[0,db])
    out, cm = multi_confu_matrix(Y_test, y_pred)
    return out, y_pred, cm[0], cm[1]


def lstm_(X_train, Y_train, X_test, Y_test, db):
    ln = len(set(Y_train.flatten()))
    # fix random seed for reproducibility
    np.random.seed(7)
    # load the dataset but only keep the top n words, zero the rest
    top_words = 5000
    train_X = X_train.reshape(-1, X_train.shape[1], 1)
    test_X = X_test.reshape(-1, X_train.shape[1], 1)
    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')
    train_X = train_X / train_X.max()
    test_X = test_X / train_X.max()
    Lab = np.concatenate((Y_train, Y_test))
    # Change the labels from categorical to one-hot encoding
    train_Y = tf.keras.utils.to_categorical(Y_train)
    test_Y = tf.keras.utils.to_categorical(Y_test)
    embedding_vecor_length = 32

    # Final evaluation of the model
    model = models.Sequential()
    model.add(layers.LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(layers.Dropout(0.2))
    # number of features on the output
    model.add(layers.Dense(ln, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()
    rnn_train = model.fit(train_X, train_Y, batch_size=128, epochs=ep, verbose=0)
    y_predict = array(np.argmax(model.predict(test_X), axis=1), axis=[3, db])
    out, cm = multi_confu_matrix(Y_test, y_predict)
    return out, y_predict, cm[0], cm[1]

def bp1_(X_train, Y_train, X_test, Y_test,db):
    ln = len(set(Y_train.flatten()))
    Lab = np.concatenate((Y_train, Y_test))
    Y_train=Y_train
    cnn_X_train = X_train.reshape((X_train.shape[0], 1, 1, X_train.shape[1]))
    cnn_X_test = X_test.reshape((X_test.shape[0], 1, 1, X_train.shape[1]))
    cnn_X_train = cnn_X_train.astype('float32')
    cnn_X_test = cnn_X_test.astype('float32')
    predict = []
    model = models.Sequential()
    input_shape=cnn_X_train[0].shape

    # TimeDistributed CNN for each timestep
    model.add(layers.TimeDistributed(layers.Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=input_shape))
    model.add(layers.TimeDistributed(layers.MaxPooling1D(pool_size=1)))
    model.add(layers.TimeDistributed(layers.Flatten()))

    # LSTM layer
    model.add(layers.LSTM(128, return_sequences=False))
    model.add(layers.Dropout(0.5))

    # Output layer
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(ln, activation='softmax' if ln > 1 else 'linear'))
    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy' if ln > 1 else 'mse',
                  metrics=['accuracy'])
    y_train =  tf.keras.utils.to_categorical(Y_train)
    model.fit(cnn_X_train, y_train, epochs=ep, batch_size=64, verbose=0)
    y_pred = array(np.argmax(model.predict(cnn_X_test), axis=1), axis=[4, db])
    out, cm = multi_confu_matrix(Y_test, y_pred)
    return out, y_pred, cm[0], cm[1]



def bp2_(x_train, y_train, x_test, Y_test, db):
    print("\nHybrid 1D CNN Model (multi-kernel concat style)")
    ln = len(set(y_train.flatten()))

    import xgboost as xgb

    xgb_classifier = xgb.XGBClassifier()
    xgb_classifier.fit(x_train, y_train)
    xgb_predictions = xgb_classifier.predict(x_test)

    # Reshape input for 1D CNN
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test  = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    input_shape_ = (x_train.shape[1], 1)

    # One-hot encode labels
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test  = tf.keras.utils.to_categorical(Y_test)

    # --- Multi-kernel convolution branches ---
    inputs = layers.Input(shape=input_shape_)

    # Branch 1: kernel size 3
    b1 = layers.Conv1D(16, kernel_size=3, activation='swish', padding='same')(inputs)
    b1 = layers.LayerNormalization()(b1)
    b1 = layers.Dropout(0.2)(b1)
    b1 = layers.MaxPooling1D(pool_size=2)(b1)

    # Branch 2: kernel size 5
    b2 = layers.Conv1D(16, kernel_size=5, activation='swish', padding='same')(inputs)
    b2 = layers.LayerNormalization()(b2)
    b2 = layers.Dropout(0.2)(b2)
    b2 = layers.MaxPooling1D(pool_size=2)(b2)

    # Branch 3: kernel size 7
    b3 = layers.Conv1D(16, kernel_size=7, activation='swish', padding='same')(inputs)
    b3 = layers.LayerNormalization()(b3)
    b3 = layers.Dropout(0.2)(b3)
    b3 = layers.MaxPooling1D(pool_size=2)(b3)

    # Concatenate learned features
    concat = layers.Concatenate()([b1, b2, b3])

    # Shared deeper conv layers
    x = layers.Conv1D(32, kernel_size=3, activation='swish', padding='same')(concat)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Dense representation
    x = layers.Flatten()(x)
    x = layers.Dense(120, activation='swish')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(84, activation='swish')(x)
    x = layers.Dropout(0.3)(x)

    # Final softmax prediction (instead of XGBoost)
    outputs = layers.Dense(ln, activation='softmax')(x)

    # Build model
    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train
    model.fit(x_train, y_train, epochs=ep, batch_size=64, validation_split=0.2, verbose=0)

    # Predict
    y_predict = np.argmax(model.predict(x_test), axis=1)
    y_predict = array(y_predict, axis=[0, db])   # your custom wrapper
    out, cm = multi_confu_matrix(Y_test, y_predict)
    return out, y_predict, cm[0], cm[1]


def efficient_net(x_train, y_train, x_test, y_test, db):
    from tensorflow.keras import layers, models
    from tensorflow.keras.optimizers import Adam

    # Reshape for 1D CNN
    x_train = np.array(x_train).reshape((x_train.shape[0], x_train.shape[1], 1)).astype('float32') / 255
    x_test = np.array(x_test).reshape((x_test.shape[0], x_test.shape[1], 1)).astype('float32') / 255
    y_train = np.array(y_train).astype('int')
    y_test = np.array(y_test).astype('int')

    num_classes = len(np.unique(y_train))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)

    # Define a simplified EfficientNet-like 1D model
    inputs = layers.Input(shape=(x_train.shape[1], 1))
    # Stem
    x = layers.Conv1D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    # MBConv-like blocks (use SeparableConv1D instead of DepthwiseConv1D)
    for filters in [32, 64, 128]:
        x = layers.SeparableConv1D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2, padding="same")(x)
    # Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    # Train
    if rn:model.fit(x_train, y_train, batch_size=128, epochs=ep, verbose=0)
    # Predict
    pred = model.predict(x_test)
    y_pred = np.argmax(pred, axis=1)
    # Example confusion matrix call (replace with your own function)
    out, cm = multi_confu_matrix(y_test, y_pred)
    return out, y_pred, cm[0], cm[1]


def lenet_(x_train, y_train, x_test, Y_test, db):
    print("\n LeNet 1D Model")
    ln = len(set(y_train.flatten()))

    # Reshape input for 1D CNN: (samples, timesteps, features)
    # Example: if x_train is (N, 1024), reshape to (N, 1024, 1)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test  = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    input_shape_ = (x_train.shape[1], 1)

    # One-hot encode labels
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test  = tf.keras.utils.to_categorical(Y_test)

    # Build the 1D LeNet model
    model = models.Sequential()
    # First Conv1D + Pooling
    model.add(layers.Conv1D(16, kernel_size=5, activation='relu', input_shape=input_shape_))
    model.add(layers.MaxPooling1D(pool_size=2))
    # Second Conv1D + Pooling
    model.add(layers.Conv1D(32, kernel_size=5, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    # Fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(84, activation='relu'))
    model.add(layers.Dense(ln, activation='softmax'))

    # Compile
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train
    model.fit(x_train, y_train, epochs=ep, batch_size=64, validation_split=0.2, verbose=0)

    # Predict
    y_predict = np.argmax(model.predict(x_test), axis=1)
    y_predict = array(y_predict, axis=[0, db])   # your custom wrapper
    out, cm = multi_confu_matrix(Y_test, y_predict)

    return out, y_predict, cm[0], cm[1]


def conv_shufflenet_(X_train, Y_train, X_test, Y_test, db):
    # Channel Shuffle for 1D
    def channel_shuffle(x, groups):
        batch_size, length, channels = tf.shape(x)[0], tf.shape(x)[1], x.shape[-1]
        channels_per_group = channels // groups
        x = tf.reshape(x, [batch_size, length, groups, channels_per_group])
        x = tf.transpose(x, [0, 1, 3, 2])
        x = tf.reshape(x, [batch_size, length, channels])
        return x

    # ShuffleNet Unit (Block) for 1D
    def shufflenet_unit(x, out_channels, groups, stride=1):
        shortcut = x
        in_channels = int(x.shape[-1])

        # 1x1 Group Convolution
        group_conv = layers.Conv1D(out_channels // 4, kernel_size=1, use_bias=False)(x)
        group_conv = layers.BatchNormalization()(group_conv)
        group_conv = layers.ReLU()(group_conv)

        # Channel Shuffle
        group_conv = channel_shuffle(group_conv, groups)

        # Depthwise Separable Convolution (1D)
        dw_conv = layers.SeparableConv1D(out_channels, kernel_size=3, strides=stride,
                                         padding='same', use_bias=False)(group_conv)
        dw_conv = layers.BatchNormalization()(dw_conv)

        # Skip connection
        if stride == 1 and in_channels == out_channels:
            output = layers.Add()([shortcut, dw_conv])
        else:
            output = dw_conv

        output = layers.ReLU()(output)
        return output

    # ShuffleNet Model (1D)
    def build_shufflenet(input_shape, num_classes, num_groups=3):
        input_layer = layers.Input(shape=input_shape)

        # Initial Convolution Layer
        x = layers.Conv1D(24, kernel_size=3, strides=2, padding='same', use_bias=False)(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

        # ShuffleNet Blocks
        x = shufflenet_unit(x, 240, groups=num_groups, stride=2)
        x = shufflenet_unit(x, 480, groups=num_groups, stride=1)
        x = shufflenet_unit(x, 960, groups=num_groups, stride=1)

        # Global Average Pooling
        x = layers.GlobalAveragePooling1D()(x)

        # Fully connected layer
        output_layer = layers.Dense(num_classes, activation='softmax')(x)

        model = models.Model(inputs=input_layer, outputs=output_layer)
        return model

    # Example usage
    input_shape = (X_train.shape[1], 1)
    num_classes = len(set(Y_train.flatten()))

    model = build_shufflenet(input_shape, num_classes, num_groups=3)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Reshape data
    x_train = np.expand_dims(X_train, axis=-1)
    x_test = np.expand_dims(X_test, axis=-1)
    y_train_cat = tf.keras.utils.to_categorical(Y_train, num_classes)

    model.fit(x_train, y_train_cat, epochs=ep, batch_size=64, verbose=0)
    y_pred = np.argmax(model.predict(x_test), axis=-1)

    y_pred = array(y_pred, axis=[0, db])
    out, cm = multi_confu_matrix(Y_test, y_pred)
    return out, y_pred, cm[0], cm[1]


def dense_net_(X_train, y_train, X_test, y_test, db):
    Input, compression = 1, 0.1
    n = [0, 1]

    # Reshape for 1D input: (samples, timesteps, channels)
    X_train = np.resize(X_train, (X_train.shape[0], X_train.shape[1], 1))  # example: 1024 timesteps
    X_test = np.resize(X_test, (X_test.shape[0], X_test.shape[1], 1))
    timesteps, channel_size = X_train.shape[1], X_train.shape[2]
    num_classes = len(np.unique(y_train))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)

    def denseblock(input, num_filter=12, dropout_rate=0.2):
        temp = input
        for _ in range(l):
            BatchNorm = layers.BatchNormalization()(temp)
            relu = layers.Activation('relu')(BatchNorm)

            Conv1D_3 = layers.Conv1D(int(num_filter * compression), 3, use_bias=False, padding='same')(relu)
            if dropout_rate > 0:
                Conv1D_3 = layers.Dropout(dropout_rate)(Conv1D_3)
            concat = layers.Concatenate(axis=-1)([temp, Conv1D_3])
            temp = concat
        return temp

    def transition(input, num_filter=12, dropout_rate=0.2):
        BatchNorm = layers.BatchNormalization()(input)
        relu = layers.Activation('relu')(BatchNorm)
        Conv1D_BottleNeck = layers.Conv1D(int(num_filter * compression), 1, use_bias=False, padding='same')(relu)
        if dropout_rate > 0:
            Conv1D_BottleNeck = layers.Dropout(dropout_rate)(Conv1D_BottleNeck)
        avg = layers.AveragePooling1D(pool_size=2)(Conv1D_BottleNeck)
        return avg

    def output_layer(input):
        BatchNorm = layers.BatchNormalization()(input)
        relu = layers.Activation('relu')(BatchNorm)
        global_average_pooling = layers.GlobalAveragePooling1D()(relu)
        global_max_pooling = layers.GlobalMaxPooling1D()(relu)
        concat_lay = layers.Concatenate(axis=-1)([global_average_pooling, global_max_pooling])
        flat = layers.Flatten()(concat_lay)
        output = layers.Dense(num_classes, activation='softmax')(flat)
        return output

    l = 7
    input = layers.Input(shape=(timesteps, channel_size))
    First_Conv1D = layers.Conv1D(30, 3, use_bias=False, padding='same')(input)

    First_Block = denseblock(First_Conv1D, 30, 0.5)
    First_Transition = transition(First_Block, 30, 0.5)
    Last_Block = denseblock(First_Transition, 30, 0.5)
    output = output_layer(Last_Block)

    model = models.Model(inputs=[input], outputs=[output])
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    # Data augmentation for 1D signals (optional, usually custom)
    # For now, skip ImageDataGenerator since it's 2D-specific

    y_predict = array(np.argmax(model.predict(X_test), axis=-1), axis=[2, db])
    out, cm = multi_confu_matrix(y_test, y_predict)
    return out, y_predict, cm[0], cm[1]


def auto_cor(prot_values, ui, k):
    # Placeholder for your autocorrelation function (eqn1 base(pdf))
    # Implement the actual formula here
    return np.correlate(prot_values - ui, prot_values - ui, mode='valid')[:k]

def input_dataread():
    db=1
    an = 0
    def sequence_base_features():
        filename = './Dataset/AT/sequence.txt'
        file = 'prot.xlsx'

        # Read Excel
        ex_file = pd.read_excel(file, header=None).values
        AA = pd.read_excel(file, header=None)

        # Read text file
        C = pd.read_csv(filename, sep="\t", header=None).values

        seq_base_feat = []
        for i in range(len(C)):
            textData = C[i, 1]
            vi = None
            for j in range(len(textData)):
                val = textData[j]
                id_vals = AA.iloc[:, 0].values
                idx = np.where(id_vals == val)[0]

                if len(idx) == 0:
                    continue
                idx = idx[0]

                if idx == 21:
                    prot_values = ex_file[idx - 1, :]
                else:
                    prot_values = ex_file[idx, :]

                ui = np.mean(prot_values, axis=0)

                Ac = []
                for k in range(1, 3):
                    Ac.append(auto_cor(prot_values, ui, k))
                Ac = np.array(Ac)

                u_mean = np.mean(Ac, axis=1)
                stan = np.std(Ac, axis=1)

                Si = (Ac - u_mean) / stan
                vi = (Si - np.min(Si)) / (np.max(Si) - np.min(Si))

            if vi is not None:
                seq_base_feat.append(vi.flatten())

        seq_base_feat = np.array(seq_base_feat)
        # np.save("saved data/seq_base_feat.npy", seq_base_feat)

    if an: sequence_base_features()

    def  gene_ontology_feat():
        # Second part
        filename1 = 'GO_FET.xlsx'
        data = pd.read_excel(filename1, header=None).values
        # raw = pd.read_excel(filename1, header=None)
        raw=data

        filenamee = './Dataset/AT/sequence.txt'
        C = pd.read_csv(filenamee, sep="\t", header=None).values

        del_id = []
        go_feat = []

        for i in range(756):
            idd = C[i, 0]  # sequence
            val = raw[:, 0]
            idx1 = np.where(val == idd)[0]

            if len(idx1) == 0:
                del_id.append(idx1)
                # np.save("del_id.npy", del_id)
            else:
                idx1 = idx1[0]
                go_feat.append(data[idx1, :])

        go_feat = np.array(go_feat)
        # np.save("saved data/go_feat.npy", go_feat)

    if an: gene_ontology_feat()


    def network_features(db):
        # ============================================================
        # PATHS
        # ============================================================

        DATASET_DIR = "./Dataset/AT"

        PAIR_FILE = os.path.join(DATASET_DIR, "AT_pair.txt")

        # ============================================================
        # 1. READ AT_pair.txt
        # ============================================================

        def read_pairs():

            df = pd.read_csv(
                PAIR_FILE,
                sep=r"\s+",
                header=None,
                names=["protein1", "protein2", "label"]
            )

            print("AT_pair shape:", df.shape)
            print(df.head())

            protein1=df['protein1'].values
            label1=df['label'].values

            go_term_txt =  pd.read_csv('Dataset/AT/go_term.txt', sep=r"\s+", header=None,)
            protein =go_term_txt.iloc[:,0].values

            num_labels = pd.Series(label1).replace({'+': 1, '-': 0}).values
            np.save('saved data/fin_label' + str(db),num_labels)
            return df

        # ============================================================
        # 2. BUILD PPI NETWORK
        # ============================================================

        def build_ppi_network(df):

            G = nx.Graph()

            # Add ALL proteins as nodes
            proteins = set(df["protein1"]) | set(df["protein2"])
            G.add_nodes_from(proteins)

            # Only confirmed interactions (+)
            positive_pairs = df[df["label"] == "+"]

            for _, row in positive_pairs.iterrows():

                p1 = row["protein1"]
                p2 = row["protein2"]

                if p1 != p2:
                    G.add_edge(p1, p2)

            print("\nPPI Network")
            print("Nodes :", G.number_of_nodes())
            print("Edges :", G.number_of_edges())

            if G.number_of_nodes() > 0:
                avg_degree = (
                        2 * G.number_of_edges()
                        / G.number_of_nodes()
                )

                print("Average degree:", avg_degree)

            return G

        # ============================================================
        # 3. COMMON NEIGHBORS
        # ============================================================

        def common_neighbors(G, p1, p2):

            n1 = set(G.neighbors(p1))
            n2 = set(G.neighbors(p2))

            return len(n1.intersection(n2))

        # ============================================================
        # 4. JACCARD INDEX
        # ============================================================

        def jaccard_index(G, p1, p2):

            n1 = set(G.neighbors(p1))
            n2 = set(G.neighbors(p2))

            intersection = n1.intersection(n2)
            union = n1.union(n2)

            if len(union) == 0:
                return 0.0

            return len(intersection) / len(union)

        # ============================================================
        # 5. ADAMIC-ADAR
        # ============================================================

        def adamic_adar(G, p1, p2):

            n1 = set(G.neighbors(p1))
            n2 = set(G.neighbors(p2))

            common = n1.intersection(n2)

            score = 0.0

            for node in common:

                degree = G.degree(node)

                if degree > 1:
                    score += 1.0 / np.log(degree)

            return score

        # ============================================================
        # 6. PREFERENTIAL ATTACHMENT
        # ============================================================

        def preferential_attachment(G, p1, p2):

            degree1 = G.degree(p1)
            degree2 = G.degree(p2)

            return degree1 * degree2

        # ============================================================
        # 7. OTSUKA-OCHIAI
        # ============================================================

        def otsuka_ochiai(G, p1, p2):

            n1 = set(G.neighbors(p1))
            n2 = set(G.neighbors(p2))

            common = len(n1.intersection(n2))

            degree1 = G.degree(p1)
            degree2 = G.degree(p2)

            denominator = np.sqrt(
                degree1 * degree2
            )

            if denominator == 0:
                return 0.0

            return common / denominator

        # ============================================================
        # 8. EXTRACT ALL 5 NETWORK FEATURES
        # ============================================================

        def extract_network_features(df, G):

            features = []

            for _, row in df.iterrows():
                p1 = row["protein1"]
                p2 = row["protein2"]

                cn = common_neighbors(
                    G, p1, p2
                )

                jac = jaccard_index(
                    G, p1, p2
                )

                aa = adamic_adar(
                    G, p1, p2
                )

                pa = preferential_attachment(
                    G, p1, p2
                )

                oo = otsuka_ochiai(
                    G, p1, p2
                )

                features.append([
                    cn,
                    jac,
                    aa,
                    pa,
                    oo
                ])

            feature_df = pd.DataFrame(
                features,
                columns=[
                    "Common_Neighbors",
                    "Jaccard",
                    "Adamic_Adar",
                    "Preferential_Attachment",
                    "Otsuka_Ochiai"
                ]
            )

            return feature_df

        # ============================================================
        # 9. MAIN
        # ============================================================

        def network_feature_extraction():

            # Read pair data
            df = read_pairs()

            # Build network from confirmed (+) interactions
            G = build_ppi_network(df)

            # Extract 5 network features
            network_features = extract_network_features(
                df,
                G
            )
            if rn: np.save('saved data/network_feat.npy',network_features)
            # Combine with original pair information
            result = pd.concat(
                [
                    df.reset_index(drop=True),
                    network_features
                ],
                axis=1
            )

            # Save
            output_file = os.path.join(
                DATASET_DIR,
                "network_features.csv"
            )

            result.to_csv(
                output_file,
                index=False
            )

            print("\n========================================")
            print("Network feature extraction completed")
            print("========================================")


            return result


        result = network_feature_extraction()

    if an:network_features(db)



def capsnet_(x_train, Y_train,  x_test,Y_test, db):

    # === CAPSNET LAYERS (UNCHANGED) ===
    class ConvLayer(nn.Module):
        def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
            super(ConvLayer, self).__init__()

            self.conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=1
                                  )

        def forward(self, x):
            return F.relu(self.conv(x))

    class PrimaryCaps(nn.Module):
        def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9):
            super().__init__()
            self.capsules = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=0)
                for _ in range(num_capsules)
            ])

        def forward(self, x):
            u = [capsule(x) for capsule in self.capsules]
            u = torch.stack(u, dim=1).view(x.size(0), 32 * 6 * 6, -1)
            return self.squash(u)

        def squash(self, input_tensor):
            squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
            return squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))

    class DigitCaps(nn.Module):
        def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16):
            super().__init__()
            self.num_routes, self.num_capsules = num_routes, num_capsules
            self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

        def forward(self, x):
            batch_size = x.size(0)
            x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)
            W = torch.cat([self.W] * batch_size, dim=0)
            u_hat = torch.matmul(W, x)
            b_ij = torch.zeros(1, self.num_routes, self.num_capsules, 1, device=x.device)
            for iteration in range(3):
                c_ij = F.softmax(b_ij, dim=2).repeat(batch_size, 1, 1, 1).unsqueeze(4)
                s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
                v_j = self.squash(s_j)
                if iteration < 2:
                    a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                    b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)
            return v_j.squeeze(1)

        def squash(self, input_tensor):
            squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
            return squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))


    # === ✅ FIXED DECODER (CRITICAL FIX) ===
    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()

            self.reconstraction_layers = nn.Sequential(
                nn.Linear(16 * 10, 512), # FC
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 784),
                nn.Sigmoid()
            )

        def forward(self, x, data):
            classes = torch.sqrt((x ** 2).sum(2))
            classes = F.softmax(classes)

            _, max_length_indices = classes.max(dim=1)
            from torch.autograd import Variable
            masked = Variable(torch.sparse.torch.eye(10))
            if USE_CUDA:
                masked = masked.cuda()
            masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)

            reconstructions = self.reconstraction_layers((x * masked[:, :, None, None]).view(x.size(0), -1))
            reconstructions = reconstructions.view(-1, 1, 28, 28)

            return reconstructions, masked


    # === COMPLETE CAPSNET ===
    class CapsNet(nn.Module):
        def __init__(self, num_classes=10):
            super(CapsNet, self).__init__()
            self.conv_layer = ConvLayer()
            self.primary_capsules = PrimaryCaps()
            self.digit_capsules = DigitCaps()
            self.decoder = Decoder()

            self.mse_loss = nn.MSELoss()

        def forward(self, data):
            output = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))
            reconstructions, masked = self.decoder(output, data)
            return output, reconstructions, masked

        def loss(self, data, x, target, reconstructions):
            return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)

        def margin_loss(self, x, labels, size_average=True):
            batch_size = x.size(0)

            v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))

            left = F.relu(0.9 - v_c).view(batch_size, -1)
            right = F.relu(v_c - 0.1).view(batch_size, -1)

            loss = labels * left + 0.5 * (1.0 - labels) * right
            loss = loss.sum(dim=1).mean()

            return loss

        def reconstruction_loss(self, data, reconstructions):
            loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1),
                                 data.view(reconstructions.size(0), -1))
            return loss * 0.0005

    # === FIXED RESHAPE ===
    def reshape_(data):
        from skimage.transform import resize
        data = np.array([resize(img, (28, 28), anti_aliasing=True) for img in data])
        return data[:, np.newaxis, :, :].astype(np.float32)

        # data = data.reshape(-1, 28, 28)
        # return data[:, np.newaxis, :, :].astype(np.float32)

    capsule_net = CapsNet().to(device)
    from torch.optim import Adam
    optimizer = Adam(capsule_net.parameters(), lr=0.001)

    X_train = reshape_(x_train)
    X_test = reshape_(x_test)

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    # Test forward pass
    test_batch = torch.FloatTensor(X_train[:4]).to(device)
    output, recon, masked = capsule_net(test_batch)
    # print(f"✅ Forward OK! Output: {output.shape}, Recon: {recon.shape}")

    # Quick training test
    batch_X = torch.FloatTensor(X_train[:16]).to(device)
    target = torch.zeros(16, 10, device=device)
    target[torch.arange(16), torch.randint(0, 10, (16,))] = 1

    optimizer.zero_grad()
    output, recon, masked = capsule_net(batch_X)
    loss = capsule_net.loss(batch_X, output, target, recon)
    loss.backward()
    optimizer.step()
    # print(f"✅ Training step OK! Loss: {loss.item():.4f}")

    # === PREDICTION FUNCTION ===
    def predict_capsnet(model, X_test, batch_size=32):
        """
        FIXED: Always reshape input to [B, 1, 28, 28] before Conv2d
        """
        model.eval()
        predictions = []

        # ✅ ALWAYS RESHAPE - whether flat or already shaped
        if len(X_test.shape) == 2 and X_test.shape[1] == 784:
            X_test = reshape_(X_test)  # [N, 784] → [N, 1, 28, 28]
        elif len(X_test.shape) != 4 or X_test.shape[1] != 1:
            X_test = reshape_(X_test.reshape(-1, 784))

        print(f"Predicting on shape: {X_test.shape}")  # Debug

        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch_X = torch.FloatTensor(X_test[i:i + batch_size]).to(device)
                print(f"Batch shape: {batch_X.shape}")  # Debug
                output, _, _ = model(batch_X)
                capsule_lengths = torch.sqrt((output ** 2).sum(2))
                batch_pred = torch.argmax(capsule_lengths, dim=1).cpu().numpy()
                predictions.extend(batch_pred)

        return np.array(predictions)

    # Get predictions
    y_pred = predict_capsnet(capsule_net, X_test)  #
    y_predict =array(y_pred.flatten(),axis=[0,db])
    out, cm = multi_confu_matrix(Y_test, y_predict)
    return out, y_predict, cm[0], cm[1]




Read=False
def full_analysis():
    db=1
    # --------- # *** STEP1: Feature Extraction *** # -------#
    if Read: input_dataread()


    # --------- # *** STEP2: Adaptive Feature Fusion*** # -------#
    def load_data():

        # -----------------------------
        x1= np.load('saved data/go_feat.npy')
        x2= np.load('saved data/seq_base_feat.npy')
        x3= np.load('saved data/network_feat.npy')
        y1=np.load('saved data/fin_label' + str(db)+'.npy')
        return x1,x2,x3,y1

    x,y,z, label =load_data()



    # ---------------- Residual Pooling Fusion ----------------
    class ResidualPoolingFusionSeq(nn.Module):
        def __init__(self, in_dim=64, out_dim=64):
            super().__init__()
            self.fc_concat = nn.Linear(in_dim * 3, out_dim)
            self.norm = nn.LayerNorm(out_dim)
            self.fc1 = nn.Linear(out_dim, out_dim)
            self.fc2 = nn.Linear(out_dim, out_dim)

        def forward(self, x):  # (batch, seq_len, d_model)
            max_pool, _ = x.max(dim=1)  # (batch, d_model)
            avg_pool = x.mean(dim=1)  # (batch, d_model)
            mixed_pool = 0.5 * (max_pool + avg_pool)

            concat = torch.cat([max_pool, avg_pool, mixed_pool], dim=-1)  # (batch, d_model*3)
            fused = self.fc_concat(concat)  # (batch, out_dim)
            fused = self.norm(fused)

            path1 = torch.sigmoid(self.fc1(fused))
            path2 = torch.sigmoid(self.fc2(fused))
            out = path1 + path2  # (batch, out_dim)
            return out.unsqueeze(1)  # (batch, 1, out_dim)

    # ---------------- Axial Cross Attention ----------------
    class AxialCrossAttention(nn.Module):
        def __init__(self, d_model=64):
            super().__init__()
            self.Q1 = nn.Linear(d_model, d_model)
            self.K1 = nn.Linear(d_model, d_model)
            self.V1 = nn.Linear(d_model, d_model)
            self.Q2 = nn.Linear(d_model, d_model)
            self.K2 = nn.Linear(d_model, d_model)
            self.V2 = nn.Linear(d_model, d_model)

        def forward(self, x):  # (batch, seq_len, d_model)
            Q1, K1, V1 = self.Q1(x), self.K1(x), self.V1(x)
            scores1 = torch.matmul(Q1, K1.transpose(-2, -1)) / torch.sqrt(
                torch.tensor(Q1.size(-1), dtype=torch.float32))
            weights1 = F.softmax(scores1, dim=-1)
            attn1 = torch.matmul(weights1, V1)

            Q2, K2, V2 = self.Q2(x), self.K2(x), self.V2(x)
            scores2 = torch.matmul(Q2, K2.transpose(-2, -1)) / torch.sqrt(
                torch.tensor(Q2.size(-1), dtype=torch.float32))
            weights2 = F.softmax(scores2, dim=-1)
            attn2 = torch.matmul(weights2, V2)

            return attn1 + attn2  # (batch, seq_len, d_model)

    # ---------------- MS-CAM ----------------
    class MSCAM(nn.Module):
        """Multiscale Channel Attention Module for 1D signals"""

        def __init__(self, channels, reduction=16):
            super().__init__()
            self.local_conv1 = nn.Conv1d(channels, channels // reduction, kernel_size=1)
            self.local_conv2 = nn.Conv1d(channels // reduction, channels, kernel_size=1)
            self.global_conv1 = nn.Conv1d(channels, channels // reduction, kernel_size=1)
            self.global_conv2 = nn.Conv1d(channels // reduction, channels, kernel_size=1)
            self.leaky_relu = nn.LeakyReLU(0.2)

        def forward(self, x):
            # x: (batch, seq_len, channels)
            x_perm = x.permute(0, 2, 1)  # (B, C, L)

            # Local branch
            local_feat = self.leaky_relu(self.local_conv1(x_perm))
            local_feat = self.local_conv2(local_feat)
            local_feat = local_feat.mean(dim=2)  # (B, C)

            # Global branch
            gap = x_perm.mean(dim=2, keepdim=True)
            g1 = self.leaky_relu(self.global_conv1(gap))
            g2 = self.global_conv2(g1)
            global_feat = g2.squeeze(-1)  # (B, C)

            # Fusion
            weights = torch.sigmoid(local_feat + global_feat).unsqueeze(1)  # (B,1,C)
            out = x * weights
            return out

    # ---------------- Fusion Pipeline ----------------
    class FusionPipeline(nn.Module):
        def __init__(self, in_dims=(421, 28, 5), d_model=64):
            super().__init__()
            self.proj_x = nn.Linear(in_dims[0], d_model)
            self.proj_y = nn.Linear(in_dims[1], d_model)
            self.proj_z = nn.Linear(in_dims[2], d_model)

            self.residual_fusion = ResidualPoolingFusionSeq(d_model, d_model)
            self.attn = AxialCrossAttention(d_model)
            self.mscam = MSCAM(d_model)

        def forward(self, x, y, z):
            # Project inputs (batch, features) → (batch, d_model)
            x = self.proj_x(x).unsqueeze(1)  # (batch,1,d_model)
            y = self.proj_y(y).unsqueeze(1)
            z = self.proj_z(z).unsqueeze(1)

            # Residual pooling fusion
            x_fused = self.residual_fusion(x)
            y_fused = self.residual_fusion(y)
            z_fused = self.residual_fusion(z)

            # Attention
            axial_cross_x = self.attn(x_fused)
            axial_cross_y = self.attn(y_fused)
            axial_cross_z = self.attn(z_fused)

            # Iterative fusion
            y_dashed = axial_cross_x + axial_cross_y + axial_cross_z
            mscam_out1 = self.mscam(y_dashed)
            add_mscam = (mscam_out1 * axial_cross_x) + (mscam_out1 * axial_cross_z)
            mscam_out2 = self.mscam(add_mscam)
            add_mscam_out = (mscam_out2 * axial_cross_x) + (mscam_out2 * axial_cross_z) + mscam_out2

            # Reduce to single vector
            final_vec = add_mscam_out.mean(dim=1)  # (batch,d_model)
            return final_vec



    # Convert to torch
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    z = torch.tensor(z, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y, z = x.to(device), y.to(device), z.to(device)

    pipeline = FusionPipeline(in_dims=(x.shape[1], y.shape[1], z.shape[1] ), d_model=64).to(device)
    output = pipeline(x, y, z)
    print("Output shape:", output.shape)
    fused_feat = output.detach().numpy()
    np.save('saved data/fused_feat'+str(db),fused_feat)


    # --------- # *** STEP3: Prediction *** # -------#

    def imp_cnn_lstm(X_train, X_test, Y_train, Y_test, db):
        ln = len(set(Y_train.flatten()))

        def fuzzy_cbam_ss(x):
            high_membership = tf.sigmoid(x)
            low_membership = tf.sigmoid(-x)
            out_fsc = high_membership * x + low_membership * (1 - x)
            out_lrelu = layers.LeakyReLU()(out_fsc)

            # ----- Channel Attention -----
            def channel_attention(x, ratio=8):
                channels = x.shape[-1]
                avg_pool = tf.reduce_mean(x, axis=1, keepdims=True)
                max_pool = tf.reduce_max(x, axis=1, keepdims=True)
                add_out = avg_pool + max_pool

                shared_mlp = tf.keras.Sequential([
                    layers.Dense(channels // ratio, activation='relu'),
                    layers.Dense(channels)
                ])

                avg_out = shared_mlp(add_out)
                max_out = shared_mlp(add_out)
                scale = tf.nn.sigmoid(avg_out + max_out)
                return x * scale

            # ----- Spatial Attention -----
            def spatial_attention(x):
                avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
                max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
                concat = tf.concat([avg_pool, max_pool], axis=-1)
                scale = layers.Conv1D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
                return x * scale
                # Channel attention

            ch_att = channel_attention(out_lrelu)
            # Spatial attention
            sp_ch_attention = spatial_attention(ch_att)

            sigma = tf.math.reduce_variance(sp_ch_attention)
            alpha = 0.6
            Thr = alpha * sigma

            def fuzzy_threshold(x, thr=0.5):
                # Case 1: x > thr → x - thr
                pos_case = tf.where(x > thr, x - thr, tf.zeros_like(x))
                # Case 2: x < -thr → x + thr
                neg_case = tf.where(x < -thr, x + thr, tf.zeros_like(x))
                # Combine both cases
                return pos_case + neg_case

            # Wrap in a Lambda layer for Keras usage
            Thr = 0.6  # example threshold
            fsh_layer = layers.Lambda(lambda t: fuzzy_threshold(t, Thr))

            fsh = fsh_layer(sp_ch_attention)

            return fsh



        # Spatial Pyramid Pooling placeholder for 1D
        def spatial_pyramid_pool(x):
            pool1 = layers.GlobalAveragePooling1D()(x)
            pool2 = layers.GlobalMaxPooling1D()(x)
            concat = layers.Concatenate()([pool1, pool2])
            return concat

        # ----- Model Definition -----
        def build_cnn_lstm_1d(input_shape, num_classes=1):
            inputs = layers.Input(shape=input_shape)

            # Conv Block 1
            x = layers.Conv1D(32, 3, padding='same')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)
            x = fuzzy_cbam_ss(x)

            # Conv Block 2
            x = layers.Conv1D(64, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('tanh')(x)
            x = fuzzy_cbam_ss(x)

            # Conv Block 3
            x = layers.Conv1D(128, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)

            # Spatial Pyramid Pool
            spp = spatial_pyramid_pool(x)
            spp = tf.expand_dims(spp, axis=1)  # make it sequence-like for LSTM
            spp = fuzzy_cbam_ss(spp)

            # LSTM
            lstm_out = layers.LSTM(64, return_sequences=False)(spp)
            lstm_out = layers.Dropout(0.5)(lstm_out)

            # Fully Connected
            fc = layers.Dense(64, activation='relu')(lstm_out)
            fc = layers.BatchNormalization()(fc)
            fc = layers.Dropout(0.5)(fc)
            fc = layers.Dense(64, activation='relu')(fc)

            outputs = layers.Dense(ln, activation='sigmoid')(fc)

            model = models.Model(inputs, outputs)
            return model

        # ----- Build & Compile -----
        input_shape = (X_train.shape[1], 1)  # Example: 500-sample ECG signal, single channel
        model = build_cnn_lstm_1d(input_shape)

        model.compile(optimizer=optimizers.Adam(1e-3),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.summary()
        y_train_cat=tf.keras.utils.to_categorical(Y_train)
        # ----- Fit -----
        model.fit(X_train, y_train_cat, epochs=ep, batch_size=128, validation_split=0.2,verbose=0)
        # ----- Prediction -----
        preds = np.argmax(model.predict(X_test),axis=1)
        return preds

    def prop_lb(X_train, X_test, Y_train, Y_test, db):
        # ######## <<<<< Classification >>>>>>>>> ########
        print('proposed CNN_LSTM \n')
        predict=imp_cnn_lstm(X_train, X_test, Y_train, Y_test, db)
        return predict


    def Testing_process(feat, label, db):
        def comp(X_train, X_test, Y_train, Y_test, db):
            clmn = ['Base1', 'Base2', 'LSTM', 'Efficientnet', 'Shufflenet', 'Lenet','Densenet', 'PROPOSED']
            pred = [bp1_,bp2_,lstm_,efficient_net,  conv_shufflenet_, lenet_, dense_net_]
            pos = len(pred)
            lp = np.load('pre_evaluated/tp.npy')
            pred_out = []
            for i in range(0, pos):
                print('lp=' + str(lp) + '_method=' + str(i))
                out, y_pred, cm, ttn = pred[i](X_train, Y_train, X_test, Y_test, db)
                pred_out.append(y_pred)
            return pred_out

        def metrices_(pred, Y_test):
            out = multi_confu_matrix(Y_test, pred)
            return out


        def method(X_train, X_test, Y_train, Y_test):
            vv = []
            vv.extend(comp(X_train, X_test, Y_train, Y_test, db))
            vv.append(prop_lb(X_train, X_test, Y_train, Y_test, db))
            return vv

        learn_percent, learning_percentage = [0.8], ['80']
        for lp, lpstr in zip(learn_percent, learning_percentage):
            X_train, X_test, Y_train, Y_test = train_test_split(feat, label, train_size=lp, stratify=label, random_state=42)
            np.save(f'pre_evaluated/Actual-' + str(lpstr) + str(db), Y_test)
            np.save('pre_evaluated/Y_test', Y_test)
            np.save('pre_evaluated/tp', lp)
            pred_va = np.array(method(X_train, X_test, Y_train, Y_test))
            if rn: np.save('pre_evaluated/Predicted-' + str(lpstr) + str(db), pred_va)
            out = np.array([metrices_(pre, Y_test) for idx, pre in enumerate(pred_va)])
            result_ = np.array([out[i][0] for i in range(len(out))])
            TPTN = np.array([out[i][1][0] for i in range(len(out))])
            CM = np.array([out[i][1][1] for i in range(len(out))])
            # bp1_, bp2_, bilstm_+xgboost, csqueezenet_, conv_shufflenet, lstm+xgboost, efficient_net_

            clmn = ['Base1', 'Base2', 'LSTM', 'Efficientnet', 'Shufflenet', 'Lenet','Densenet', 'PROPOSED']
            indx = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f_measure', 'mcc', 'npv', 'fpr', 'fnr']
            globals()['df' + lpstr] = pd.DataFrame(result_.transpose(), columns=clmn, index=indx)
            indx1 = ['TP', 'TN', 'FP', 'FN']
            globals()['df_' + lpstr] = pd.DataFrame(TPTN.transpose(), columns=clmn, index=indx1)
            if rn: np.save('pre_evaluated/cm-' + str(lpstr) + str(db), CM)
        key = [ '80']
        frames = [df_80]
        df1 = pd.concat(frames, keys=key, axis=0)
        if rn: df1.to_csv(f'pre_evaluated/TP_TN-' + str(db) + '.csv')
        key = ['80']
        frames = [ df80]
        df1 = pd.concat(frames, keys=key, axis=0)
        if rn: df1.to_csv(f'pre_evaluated/Performance-' + str(db) + '.csv')

    Testing_process(fused_feat, label, db)

    ###### ablation study ###
    def kfold_validation(FEAT, Lab):


        print('K-fold validation')
        perf_A = []
        FEAT = FEAT[:, :]
        from sklearn.model_selection import KFold
        # for a in range(4):
        kfold = KFold(n_splits=5, random_state=1, shuffle=True)
        print(kfold)
        perf_A1 = []
        fold = 1
        np.save('pre_evaluated/tp', 0.7)
        # comp_pred = [a1, a2, a3, a4, a5, a6, a7, prop_lb]
        comp_pred = [bp1_,bp2_,lstm_,efficient_net,  conv_shufflenet_, lenet_, dense_net_,prop_lb]

        for train, test in kfold.split(FEAT):
            print('kfold', fold)
            X_train, X_test, Y_train, Y_test = FEAT[train], FEAT[test], Lab[train], Lab[test]
            np.save('pre_evaluated/Y_test', Y_test)

            pred_out = []
            for i in range(0, len(comp_pred)):
                if i != len(comp_pred) - 1:
                    out, y_pred, cm, ttn = comp_pred[i](X_train, Y_train, X_test, Y_test, db)
                else:
                    pred1 = comp_pred[i](X_train, X_test, Y_train, Y_test, db)  #######prop
                    out, cm = multi_confu_matrix(Y_test, pred1)
                pred_out.append(out[0])
            perf_A1.append(pred_out)
            fold += 1
        if rn: np.save('saved data/perf_A_KF' + str(db), perf_A1)

    kfold_validation(fused_feat, label)



    def ab_comp(db):
        feat = np.load('saved data/fused_feat' + str(db) + '.npy', allow_pickle=True)
        feat_o = np.load('saved data/go_feat'+ '.npy')
        feat1 = np.load('saved data/network_feat'+ '.npy')
        feat2 = np.load('saved data/seq_base_feat'+ '.npy')
        label = np.load('saved data/fin_label' + str(db) + '.npy')
        f, f_ = [feat, feat1, feat1, feat_o, feat2, feat1], 'c'

        label1 = [label, label, label, label, label]
        lp1 = [0.8]
        met1, lab1, cm1 = [], [], []
        for lp in lp1:
            c1 = []
            for i in range(0, len(label1)):
                print('LP >>>>>', lp, i)
                fet = f[i]
                fet[np.where(np.isnan(fet))] = 0.1
                fet1 = fet / np.max(fet)
                X_train, X_test, Y_train, Y_test = train_test_split(fet1, label1[i][:len(fet1)], test_size=0.33,
                                                                    random_state=0)
                np.save('pre_evaluated/Y_test', Y_test)
                np.save('pre_evaluated/tp', lp)
                vv = prop_lb(X_train, X_test, Y_train, Y_test, db)
                c1.append(multi_confu_matrix(Y_test, vv))

            comp = np.array(c1)
            vall, clasify_roc, cmm = [], [], []
            for jj in range(len(comp)):
                vall.append(comp[jj][0])
                cmm.append(comp[jj][1])
            class_val, roc_comp = [], []
            for kk in range(len(vall)):
                class_val.append(vall[kk])
            class_val = np.array(class_val)
            met1.append(vall)
            lab1.append(class_val)
            cm1.append(cmm)
        out_com = np.array(met1)[0, :, :] * hr()
        clmn = ['without adaptive feature fusion', 'with gene ontology feature', 'network based feature',
                'with sequence based feature', 'conventional cnn_lstm']
        indx = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f_measure', 'mcc', 'npv', 'fpr', 'fnr']
        import pandas as pd
        val = np.array(out_com)
        d = pd.DataFrame(val.transpose(), columns=clmn, index=indx)
        if rn == 1: d.to_csv(f'pre_evaluated/ablation_study' + str(db) + '.csv')
        return d

    if rn: ab_comp(db)

    # --------- # *** STEP4: XAI *** # -------#
    from x_AI_analysis import lime_explainable_AI
    lime_explainable_AI()



######### fully run ::cc ########

db=1
from other import popup
vvv = popup.PopupYesNo('')
if vvv == "Full analysis + Plots":
    full_analysis()
    plotresult()
    plt.show()
else:
    plotresult()
    plt.show()




#
# full_analysis()
# plotresult()
# plt.show()