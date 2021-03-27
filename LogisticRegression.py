from ProcessDataClassLR import ProcessData
from WordEmbedding import WordEmbedding
from LogisticRegressionClass import MyLogisticRegression
from sklearn.metrics import classification_report
import pickle


if __name__ == '__main__':
    #Initialize
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    labels_test_path = "data/solution.csv"
    MAX_ITER = 2500
    LEARNING_RATE = 0.00001
    SAVE_WORD2VEC = True
    SAVE_MODEL = True

    #Read data from file
    process_data = ProcessData()
    x_train, y_train = process_data.read_train_csv(train_path)
    x_test, y_test = process_data.read_test_csv(test_path, labels_test_path)
    print(x_test[0])

    #Word2Vec
    if SAVE_WORD2VEC:
        word_embedding = WordEmbedding()
        x_train_embedded = word_embedding.doc2vec(x_train)
        x_test_embedded = word_embedding.fit_transform(x_test)
        with open("all_words.pickle", "wb") as handle:
            pickle.dump([x_train_embedded, x_test_embedded], handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open("all_words.pickle", "rb") as handle:
            x_train_embedded, x_test_embedded = pickle.load(handle)

    #Logistic Regression
    if SAVE_MODEL == True:
        clf = MyLogisticRegression(max_iter=MAX_ITER, lr=LEARNING_RATE)
        total_lost = clf.fit(x_train_embedded, y_train)
        with open("model.pickle", "wb") as handle:
            pickle.dump([clf, total_lost], handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open("model.pickle", "rb") as handle:
            clf, total_lost = pickle.load(handle)

    #Predict and Evaluation Model
    y_pred = clf.predict(x_train_embedded)
    acc = clf.accuracy_score(y_train, y_pred)
    print(classification_report(y_train, y_pred))
    print(f'accuracy of training with {MAX_ITER} epochs and {LEARNING_RATE} learning rate = ', acc)

    #Test set
    y_pred = clf.predict(x_test_embedded)
    acc = clf.accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print(f'accuracy of testing with {MAX_ITER} epochs and {LEARNING_RATE} learning rate = ', acc)

    clf.plot_loss(total_lost)
    print("End program!!")