from collections import defaultdict
import time
import math
import pickle


class TwitterCredentials(object):

    def __init__(self):
        '''
        Contains Twitter account information.

        Name: Harold Tweetbot
        Account: ml697697
        Email: ml697697@gmail.com
        Password: machinelearningiscool
        '''
        # Define Twitter credentials for the Twitter API.
        self.consumer_key = 'XjJyZhSCnah1VQmJ56jZ1D3YB'
        self.consumer_secret = 'OwnbJI2u6IPZjhUD3X5GevAn54OuwO526sOX4ljgcEsFI9su18'
        self.access_token = '938534384801583104-lr029cwbvXYkXasppXOiW8O9FCunLiA'
        self.access_token_secret = 'aV9Nwg6VvtpMFBXzlCX2q4TRcC9UvzT4VfpJLX7NqRLk7'
        self.owner = 'ml697697'
        self.owner_id = '938534384801583104'


class TweetData(object):

    def __init__(self, output_filename=None, pickle_filename=None):
        """
        Stores and organizes tweet information.
        """
        self.tweet_count = 0
        if pickle_filename != None:
            self.tweets = self.load_tweets(pickle_filename)
        else:
            self.tweets = defaultdict(dict)
        self.output_filename = output_filename
        self.current_year = time.time()/60.0/60.0/24.0/30.4375/12.0 + 1970
        self.current_month = 12.0 * (self.current_year - math.floor(self.current_year)) + 1
        self.current_day = 30.4375 * (self.current_month - math.floor(self.current_month)) + 1
        self.current_time = math.floor(self.current_day + 30.4375*math.floor(self.current_month)
                                       + 365.25*math.floor(self.current_year))  # in days

    def add_tweet(self, tweet_data):
        """
        Adds tweet to dictionary and formats it if not a retweet.
        """
        if self.check_if_retweet(tweet_data) == True:
            return True
        else:
            # Add tweet data to the dictionary.
            self.tweets[self.tweet_count]['screen_name'] = tweet_data.user.screen_name
            self.tweets[self.tweet_count]['id'] = tweet_data.id
            self.tweets[self.tweet_count]['followers'] = tweet_data.user.followers_count  # followers at current time
            self.tweets[self.tweet_count]['text'] = tweet_data.full_text
            self.tweets[self.tweet_count]['hashtags'] = tweet_data.entities
            self.tweets[self.tweet_count]['time_ratio'] = self.tweet_time(tweet_data)
            self.tweets[self.tweet_count]['favorites'] = tweet_data.favorite_count
            self.tweets[self.tweet_count]['retweets'] = tweet_data.retweet_count
            self.tweets[self.tweet_count]['media'] = tweet_data.entities.has_key('media')
            self.tweets[self.tweet_count]['quoted'] = tweet_data.is_quote_status
            self.tweets[self.tweet_count]['reply'] = (tweet_data.in_reply_to_status_id != None)

            # Increment the tweet counter.
            self.tweet_count = self.tweet_count + 1
            return False

    def check_if_retweet(self, tweet_data):
        """
        Checks if tweet is a retweet.
        """
        return tweet_data._json.has_key('retweeted_status')

    def tweet_time(self, tweet_data):
        """
        Calculates the ratio of tweet age to account age.
        """
        # Time of user's account creation.
        user_day = tweet_data.user.created_at.day
        user_month = tweet_data.user.created_at.month
        user_year = tweet_data.user.created_at.year
        user_time = math.floor(user_day + 30.4375*user_month + 365.25*user_year)

        # Time of tweet.
        tweet_day = tweet_data.created_at.day
        tweet_month = tweet_data.created_at.month
        tweet_year = tweet_data.created_at.year
        tweet_time = math.floor(tweet_day + 30.4375*tweet_month + 365.25*tweet_year)

        # Calculate the ratio.
        return (tweet_time - user_time)/(self.current_time - user_time)

    def save_tweets(self):
        """
        Save the dictionary of tweet data to a pickle file.
        """
        if self.output_filename == None:
            return False
        else:
            # Open pickle file.
            pickle_out = open("%s.pickle" % self.output_filename, "wb")

            # Dump dictionary of tweets into the pickle file.
            pickle.dump(self.tweets, pickle_out)

            # Close the pickle file.
            pickle_out.close()
            return True

    def load_tweets(self, pickle_filename):
        """
        Load the dictionary of tweet data from a pickle file.
        """
        # Open pickle file.
        pickle_in = open("%s.pickle" % pickle_filename, "rb")

        # Populate the dictionary
        return pickle.load(pickle_in)
