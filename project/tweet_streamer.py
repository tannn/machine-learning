from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json

access_token = "941707847238373376-R4517e2MlbF1T3kGSOnRJmA2ay0MGZy"
access_token_secret = "wBQklX4XgWaLGXRw7kWW9mxSpAA4wyQF1VIdHnXzP94WF"
consumer_key = "z7d23cQWW3IE9IMFOV5DwjDzm"
consumer_secret = "COUn7QLcCDUj8ct8DLxY9kEjcfkOC3FNqXEr9q0x8hmOhyZrJE"

class StdOutListener(StreamListener):
    def on_data(self, data):
        decoded = json.loads(data)
        if not decoded['text'].startswith('RT'):
            try:
                with open('dataMining_190326.json', 'a') as f:
                    f.write(data)
                    return True 
            except BaseException as e:
                print("Error on_data: %s" % str(e))
        return True

    def on_error(self, status):
        print(status)


if __name__ == '__main__':
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)
 
    stream.filter(track=['Abortion', 'abortion', 'Pro-life', 'pro-life', 'Pro-choice', 'pro-choice', 'pro life', 'pro choice'
	, 'Pro life', 'Pro choice'])
