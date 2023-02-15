import utils
import probe
import random
import torch
from sklearn.feature_extraction.text import CountVectorizer
from vader import SentimentIntensityAnalyzer
import pandas as pd
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }



def get_simple_lexicon_scores(category, year, stars, lexicon, size):
    df = utils.get_star_by_year_df(category, year, stars)
    vocab, score_arr = probe.create_vocab(lexicon)
    df = df.tail(int(df.shape[0] / 2))
    test = df.sample(n=size, random_state=42)
    vectorizer = CountVectorizer(ngram_range=(1, 1), token_pattern=r'\b\w+\b', vocabulary=vocab, binary=False)
    feature_matrix = vectorizer.fit_transform(test['reviews'].values.astype('U')).toarray()
    sent = feature_matrix.dot(score_arr)

    correct = 0
    zero = 0
    for i in range(size):
        if stars == 1:
            if sent[i] < 0:
                correct += 1
        else:
            if sent[i] >= 0:
                correct += 1
        if sent[i] == 0:
            zero += 1

    return correct/size



def get_enhanced_lexicon_scores(category, year, stars, lexicon, size):
    df = utils.get_star_by_year_df(category, year, stars)
    df = df.tail(int(df.shape[0]/2))
    test = df.sample(n=size, random_state=42)
    reviews = test["reviews"].tolist()
    sent = []
    analyzer = SentimentIntensityAnalyzer()
    for r in reviews:
        try:
            vs = analyzer.polarity_scores(r)
            score = vs['compound']
            sent.append(score)
        except:
            sent.append(random.randint(-1,1))

    correct = 0
    zero = 0
    for i in range(size):
        if stars == 1:
            if sent[i] < 0:
                correct += 1
        else:
            if sent[i] >= 0:
                correct += 1
        if sent[i] == 0:
            zero += 1

    return correct/size


def get_bert_splits(category, year, stars, size):
    neg_df =  utils.get_star_by_year_df(category, year, stars[0])
    pos_df = utils.get_star_by_year_df(category, year, stars[1])
    neg_df = neg_df.tail(int(neg_df.shape[0]/2))
    pos_df = pos_df.tail(int(pos_df.shape[0]/2))
    neg_df_test = neg_df.sample(n=size, random_state=42)
    pos_df_test = pos_df.sample(n=size, random_state=42)
    neg_df_test = neg_df_test.dropna(subset=["reviews"])
    pos_df_test = pos_df_test.dropna(subset=["reviews"])
    test = pd.concat([neg_df_test,pos_df_test])
    test_labels = [0]*neg_df_test.shape[0] + [1]*pos_df_test.shape[0]
    test_reviews = test["reviews"].tolist()
    neg_df = neg_df.head(int(neg_df.shape[0]//4))
    pos_df = pos_df.head(int(pos_df.shape[0]//4))

    neg_df_test = neg_df.sample(n=size//4, random_state=42)
    pos_df_test = pos_df.sample(n=size//4, random_state=42)
    neg_df_test = neg_df_test.dropna(subset=["reviews"])
    pos_df_test = pos_df_test.dropna(subset=["reviews"])
    test = pd.concat([neg_df_test,pos_df_test])
    train_labels = [0]*neg_df_test.shape[0] + [1]*pos_df_test.shape[0]
    train_reviews = test["reviews"].tolist()
    return train_reviews, train_labels, test_reviews, test_labels


def get_bert_scores(category, year, stars, size):
    if len(year) == 1:
        train_reviews, train_labels, test_reviews, test_labels = get_bert_splits(category, year[0], stars, size)
    else:
        span = year[1] - year[0] + 1
        train_reviews, train_labels, test_reviews, test_labels = [], [], [], []
        for i in range(span):
            train_reviews_temp, train_labels_temp, test_reviews_temp, test_labels_temp = \
                get_bert_splits(category, year[0] + i, stars, size//span)
            train_reviews.extend(train_reviews_temp)
            train_labels.extend(train_labels_temp)
            test_reviews.extend(test_reviews_temp)
            test_labels.extend(test_labels_temp)



    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(train_reviews, truncation=True, padding=True)
    test_encodings = tokenizer(test_reviews, truncation=True, padding=True)
    train_dataset = ReviewDataset(train_encodings, train_labels)
    test_dataset = ReviewDataset(test_encodings, test_labels)
    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
    )
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        #eval_dataset=test_dataset,  # evaluation dataset
        compute_metrics = compute_metrics
    )

    trainer.train()
    #metrics = trainer.evaluate()

    metrics = trainer.predict(test_dataset)
    print(metrics[2])
    return metrics[2]["test_accuracy"]


if __name__ == '__main__':
    categories = ["Amazon", "Yelp", "IMDB"]
    start_years = [2000, 2007, 2000]
    end_years = [2018, 2020, 2020]
    sen = [[1, 5], [1, 5], [1, 10]]
    lexicon = utils.load_lexicon("data/vader_lexicon.txt")
    size = 2000


    print("################# basic lexicon #################")
    for i in range(3):
        results = 0
        for j in range(2):
            results += get_simple_lexicon_scores(categories[i], start_years[i], sen[i][j], lexicon, size)
        print("the accuracy for {} in year {} is {}".format(categories[i], start_years[i], round(results/2,3)))
        results = 0
        for j in range(2):
            results += get_simple_lexicon_scores(categories[i], end_years[i], sen[i][j], lexicon, size)
        print("the accuracy for {} in year {} is {}".format(categories[i], end_years[i], round(results/2,3)))
        results = 0
        span = end_years[i]-start_years[i] + 1
        for t in range(span):
            for j in range(2):
                results += get_simple_lexicon_scores(categories[i], start_years[i] + t, sen[i][j], lexicon,
                                                       size//span)
        print("the accuracy for {} overall is {}".format(categories[i], round(results /(2*span), 3)))
    print("################# enhanced lexicon #################")
    for i in range(3):
        results = 0
        for j in range(2):
            results += get_enhanced_lexicon_scores(categories[i], start_years[i], sen[i][j], lexicon, size)
        print("the accuracy for {} in year {} is {}".format(categories[i], start_years[i], round(results/2,3)))
        results = 0
        for j in range(2):
            results += get_enhanced_lexicon_scores(categories[i], end_years[i], sen[i][j], lexicon, size)
        print("the accuracy for {} in year {} is {}".format(categories[i], end_years[i], round(results/2,3)))
        results = 0
        span = end_years[i]-start_years[i] + 1
        for t in range(span):
            for j in range(2):
                results += get_enhanced_lexicon_scores(categories[i], start_years[i] + t, sen[i][j], lexicon,
                                                       size//span)
        print("the accuracy for {} overall is {}".format(categories[i], round(results /(2*span), 3)))



    print("#################       BERT       #################")
    for i in range(1):
        results = get_bert_scores(categories[i], [start_years[i]], sen[i], size)
        print("the accuracy for {} in year {} is {}".format(categories[i], start_years[i], round(results, 3)))

        results = get_bert_scores(categories[i], [end_years[i]], sen[i], size)
        print("the accuracy for {} in year {} is {}".format(categories[i], end_years[i], round(results, 3)))

        results = get_bert_scores(categories[i], [start_years[i], end_years[i]], sen[i], size)
        print("the accuracy for {} overall is {}".format(categories[i], round(results, 3)))

