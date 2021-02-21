## Part 1 - Project Description

Topic sentiment analysis for tweets.

We managed to improve the Bag of Words, we improved the model with neural networks and Word2Vec. We achieved 76% accuracy. Using an LDA model we clustered the tweets into topics. 

Business case: users will be able to get insights about what they will post to improve their tweets

## Part 2 - Technical Description

Topic Modeling:

1. Bag of words:

    I used Gensim’s Dictionary constructor to give each word in the tweet corpus a unique integer identifier.

    ```python
    from gensim.corpora import Dictionary

    text_dict = Dictionary([tweet.split() for tweet in df_train['tweet']])
    text_dict.token2id
    tweets_bow = [text_dict.doc2bow(tweet.split()) for tweet in df_train['tweet']]
    ```

    2. Fitting the LDA Model

    ```python
    from gensim.models.ldamodel import LdaModel
    k = 5
    tweets_lda = LdaModel(tweets_bow,
                          num_topics = k,
                          id2word = text_dict,
                          random_state = 1,
                          passes=10)

    tweets_lda.show_topics()
    ```

    Result: 

    ```python
    (0, 
    '0.027*"love" + 0.021*"u" + 0.017*"thanks" + 0.015*"you" + 0.010*"good" + 0.009*"say" + 0.008*"guy" + 0.007*"great" + 0.007*"hey" + 0.007*"see"'),
    ```

    The output represents 5 topics, consisting of the top keywords and associated weightage contribution to the topic. For example for this cluster we can see that the most important words for this topic are: love, u, thanks, you, good, great, hey, see 

Sentiment Analysis:

1. Built an embedding layer using Word2Vec. Tweets padded to length 100 (as average length was 45, we felt like using more than 100 would not be ideal). 

```python
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train.tweet)
sequences_train = tokenizer.texts_to_sequences(df_train.tweet)
sequences_test = tokenizer.texts_to_sequences(df_test.tweet)
X_train_seq = sequence.pad_sequences(sequences_train, maxlen=100, value=0)
X_test_seq = sequence.pad_sequences(sequences_test, maxlen=100, value=0)

for word, idx in list(w_index.items()):
    if word in list(model.wv.vocab.keys()):
        embedding_m[idx] = model.wv[word]
```

Note: Although we could have used a pretrained embedding, it is interesting to see how our results are appropriate for tweets. Example: 





2. Built a NN architecture to predict sentiment

The first layer is a non-trainable embedding layer. The weights are the embedding matrix weights.

Then, we used Dropout layers as regularization parameters (especially because we trained with a subset of the real data as it was taking too long, these dropouts were important to avoid overfitting)

We then used Bidirectional LTSM layers with 128 units. 

Finally, a Dense layer with sigmoid activation for the binary classification

```python
nn_model = Sequential()
emb_layer = Embedding(vocab_size, 200, weights=[embedding_m], input_length=100, trainable=False)
nn_model.add(emb_layer)
nn_model.add(Dropout(rate=0.4))
nn_model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
nn_model.add(Dropout(rate=0.4))
nn_model.add(Bidirectional(LSTM(units=128, return_sequences=False)))
nn_model.add(Dense(units=1, activation='sigmoid'))
nn_model.add(Dense(1, activation='sigmoid'))
nn_model.compile(loss='binary_crossentropy',optimizer="adam",metrics=['accuracy'])

nn_model.summary()
callbacks = [EarlyStopping(monitor='val_accuracy', patience=0)]
nn_model.fit(X_train_seq, y_train, batch_size=128, epochs=12, validation_split=0.2, callbacks=callbacks)
```

- Although we had reached simillar accuracies with the logistic regression model, the NN model was way more consistent. With the logistic regression, every run through would give very different results. Also, the CountVectorizer for the logreg was problematic due to its runtime, this was improved with the NN.
- This model will still be improved, we are looking into using other layers and methods of preprocessing for improval, but for now, it is providing satisfactory results. On kaggle, the highest accuracy we found for this specific dataset was of 79%, which is just 2% higher than our results.

## Part 3 - Challenges

1. Runtime: training the model on all the dataset was taking way too long. Our solution was to reduce the size of the training set. 
2. Analyzing hashtags did not give good results so we focused on topic modelling. 
3. Tring more complex models and fine-tune their hyperparameters.

## Part 4 - Remaining steps

1. Reduce the scope of the business problem
2. Now the sentiment analysis and the topic modelling are two separate models. We need to merge the results.  
3. Create an understandable output for the user.
4. Deployment
5. Preparing presentation
6. Write-up
