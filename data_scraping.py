import tweepy
import numpy as np
import sys


class TwitterCredentials(object):
    def __init__(self):
        '''
        Twitter Account Information:
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
    def __init__(self):
        # Stores and organizes tweet information.
        self.tweet = []
        self.id = []
        self.user_name = []
        self.likes = []
        self.retweets = []
        self.followers = []

    def add_tweet(self, tweet_data):

        self.tweet.append(tweet_data._json['full_text'])
        self.id.append(tweet_data._json['id'])
        self.user_name.append(tweet_data._json['user']['screen_name'])
        self.likes.append(tweet_data._json['favorite_count'])
        self.retweets.append(tweet_data._json['retweet_count'])
        self.followers.append(tweet_data._json['user']['followers_count'])


def main(user_names=["realDonaldTrump"]):
    # Scrapes tweets from specified users.

    # Instantiate your Twitter credentials.
    my_credentials = TwitterCredentials()

    # Authorize access to the Twitter API.
    auth = tweepy.OAuthHandler(my_credentials.consumer_key, my_credentials.consumer_secret)
    auth.set_access_token(my_credentials.access_token, my_credentials.access_token_secret)
    api = tweepy.API(auth)

    # Initializes a list for the scraped tweets.
    tweet_data = TweetData()

    # Iterate over all users in the list to pull tweets.
    for user_name in user_names:

        # Initialize user tweet list.
        user_tweets = []
        # Scrape the first set of user tweets (only 200 at a time).
        new_tweets = api.user_timeline(screen_name=user_name, count=200, tweet_mode='extended')

        # Save the tweets.
        user_tweets.extend(new_tweets)

        # Point to the next tweet to be scraped.
        tweet_ptr = user_tweets[-1].id - 1

        print(len(user_tweets))

        # Continue scraping tweets until depleted.
        while len(new_tweets) > 0:

            # Scrape the next set of user tweets starting at the pointer.
            new_tweets = api.user_timeline(screen_name=user_name, count=200, max_id=tweet_ptr, tweet_mode='extended')

            # Save the tweets.
            user_tweets.extend(new_tweets)

            # Point to the next tweet to be scraped.
            tweet_ptr = user_tweets[-1].id - 1
            print len(user_tweets)

        # Extract relevant data from the user tweets and transfer it to the tweet data dictionary.
        for tweet in user_tweets:
            tweet_data.add_tweet(tweet)

    return "finished"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print main()
    elif len(sys.argv) == 2:
        print main(sys.argv[1])
    else:
        sys.exit("error: incorrect number of arguments passed")