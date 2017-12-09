import tweepy
import sys
from twitter_classes import TwitterCredentials, TweetData


def main(user_names=["realDonaldTrump"], filename="twitter_data"):
    # Scrapes tweets from specified users.

    # Instantiate your Twitter credentials.
    my_credentials = TwitterCredentials()

    # Authorize access to the Twitter API.
    auth = tweepy.OAuthHandler(my_credentials.consumer_key, my_credentials.consumer_secret)
    auth.set_access_token(my_credentials.access_token, my_credentials.access_token_secret)
    api = tweepy.API(auth)

    # Initializes a list for the scraped tweets.
    tweet_data = TweetData(filename)

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

        # Continue scraping tweets until depleted.
        while len(new_tweets) > 0:

            # Scrape the next set of user tweets starting at the pointer.
            new_tweets = api.user_timeline(screen_name=user_name, count=200, max_id=tweet_ptr, tweet_mode='extended')

            # Save the tweets.
            user_tweets.extend(new_tweets)

            # Point to the next tweet to be scraped.
            tweet_ptr = user_tweets[-1].id - 1

        # Extract relevant data from the user tweets and transfer it to the tweet data dictionary.
        for tweet in user_tweets:
            tweet_data.add_tweet(tweet)

    # Save the tweets as a pickle file.
    if tweet_data.save_tweets():
        return 1
    else:
        return 0


if __name__ == "__main__":

    # Choose arguments to pass.
    if len(sys.argv) == 1:
        return_arg = main()
    elif len(sys.argv) == 3:
        return_arg = main(sys.argv[1:])
    else:
        sys.exit("error: incorrect number of arguments passed")

    # Indicate success or failure.
    if return_arg:
        sys.exit("success: tweet data was saved")
    else:
        sys.exit("error: tweet data failed to save")
