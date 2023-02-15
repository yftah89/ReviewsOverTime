import pandas as pd
import gzip
import pickle
import json
import probe
import os


# This class is designed to extract the data out of Julian McAuley dataset
class DataProvider:

    def __init__(self, category):
        self.category = category

    def get_all_data_points(self):
        df = DataProvider.get_df(self.data_path)
        return df

    @staticmethod
    def parse_from_gzip(path):
        # for memory issues
        print(path)
        max_extract = 100000000
        count = 0
        g = gzip.open(path, 'r')
        for l in g:
            if count < max_extract:
                yield json.loads(l.decode('utf-8'))
            else:
                return
            count += 1

    @staticmethod
    def get_df(path):
        i = 0
        df = {}
        for d in DataProvider.parse_from_gzip(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')


class ReviewDataProvider(DataProvider):
    def __init__(self, category):
        super().__init__(category)
        self.data_path = "data/{}.json.gz".format(self.category)

    def prepare_clean_df(self, df):
        reviews = df["reviewText"].tolist()
        scores = df["overall"].tolist()
        dates = df["reviewTime"].tolist()
        reviewers = df["reviewerID"].tolist()
        votes = df["vote"].tolist()
        subjects = df["asin"]
        years = []
        for i in range(len(dates)):
            years.append(int(dates[i].split(",")[-1]))
        probe.get_histogram(years, "years", "#reviews", "review per year", self.category)
        data = {'reviews': reviews, 'scores': scores, 'years': years, 'votes': votes, 'reviewers': reviewers,
                'subjects': subjects}
        df = pd.DataFrame(data)
        return df

    def construct_dataset(self):
        filename = "data/{}_clean_df".format(self.category)
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                clean_df = pickle.load(f)
        else:
            df = self.get_all_data_points()
            clean_df = self.prepare_clean_df(df)
            with open(filename, 'wb') as f:
                pickle.dump(clean_df, f)
        return clean_df


class YelpDataProvider:
    def __init__(self, category, data_path):
        self.category = category
        self.data_path = data_path

    def prepare_clean_df(self, df):
        reviews = df["text"].tolist()
        scores = df["stars"].tolist()
        df['year'] = df['date'].dt.year
        years = df['year'].tolist()
        reviewers = df["user_id"].tolist()
        votes = df["useful"].tolist()
        subjects = df["business_id"]
        probe.get_histogram(years, "years", "#reviews", "review per year", self.category)
        data = {'reviews': reviews, 'scores': scores, 'years': years, 'votes': votes, 'reviewers': reviewers,
                'subjects': subjects}
        df = pd.DataFrame(data)
        return df

    def construct_dataset(self):
        filename = "data/{}_clean_df".format(self.category)
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                clean_df = pickle.load(f)
        else:
            with open(self.data_path, 'rb') as f:
                df = pickle.load(f)
            clean_df = self.prepare_clean_df(df)
            with open(filename, 'wb') as f:
                pickle.dump(clean_df, f)
        return clean_df


class IMDBDataProvider:
    def __init__(self, category, data_path):
        self.category = category
        self.data_path = data_path

    def prepare_clean_df(self, df):
        reviews = df["review_detail"].tolist()
        df['rating'] = df['rating'].astype(int)
        scores = df["rating"].tolist()
        dates = df["review_date"].tolist()
        reviewers = df["reviewer"].tolist()
        votes = df["helpful"].tolist()
        subjects = df["movie"].tolist()
        positive_votes = []
        overall_votes = []
        years = []
        for i in range(len(dates)):
            years.append(int(dates[i].split()[-1]))
            positive_votes.append(votes[i][0])
            overall_votes.append(votes[i][1])
        probe.get_histogram(years, "years", "#reviews", "review per year", self.category)
        data = {'reviews': reviews, 'scores': scores, 'years': years, 'votes': positive_votes, 'reviewers': reviewers,
                'subjects': subjects, 'overall-votes':overall_votes}
        df = pd.DataFrame(data)
        return df

    def construct_dataset(self):
        filename = "data/{}_clean_df".format(self.category)
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                clean_df = pickle.load(f)
        else:
            with open(self.data_path, 'rb') as f:
                df = pickle.load(f)
                df = df.fillna(0)
                print("size = ", df.shape)
                df = df[df["review_date"] != 0]
                print("size = ", df.shape)
            clean_df = self.prepare_clean_df(df)
            with open(filename, 'wb') as f:
                pickle.dump(clean_df, f)
        return clean_df


class BookingDataProvider:
    def __init__(self, category, data_path):
        self.category = category
        self.data_path = data_path

    def prepare_clean_df(self, df):
        df['rating'] = df['Reviewer_Score'].astype(int)
        scores = df["rating"].tolist()
        dates = df["Review_Date"].tolist()
        subjects = df["Hotel_Name"].tolist()
        positive_rev = df["Positive_Review"].tolist()
        negative_rev = df["Negative_Review"].tolist()
        tags = df["Tags"].tolist()
        years = []
        reviews = []
        devices = []
        mobile_count = 0
        pc_count = 0
        for i in range(len(dates)):
            years.append(int(dates[i].split("/")[-1]))
            neg = negative_rev[i]
            if neg == "No Negative":
                neg = ""
            pos = positive_rev[i]
            if pos == "No Positive":
                pos = ""
            review = neg + " " + pos
            #scores[i] = 10

            reviews.append(review)
            if ' Submitted from a mobile device ' in tags[i]:
                devices.append("mobile")
                mobile_count += 1
            else:
                devices.append("pc")
                pc_count += 1
        print("pc = {}, mobile = {}".format(pc_count, mobile_count))
        probe.get_histogram(years, "years", "#reviews", "review per year", self.category)
        data = {'reviews': reviews, 'scores': scores, 'years': years, 'subjects': subjects, 'devices': devices}
        df = pd.DataFrame(data)
        return df

    def construct_dataset(self):
        filename = "data/{}_clean_df".format(self.category)
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                clean_df = pickle.load(f)
        else:
            df = pd.read_csv(self.data_path)
            print("size = ", df.shape)
            clean_df = self.prepare_clean_df(df)
            with open(filename, 'wb') as f:
                pickle.dump(clean_df, f)
        return clean_df







