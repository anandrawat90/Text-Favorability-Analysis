from re import sub, compile
from twython import Twython
import emoji
import Properties

APP_KEY = Properties.APP_KEY  # consumer keys
APP_SECRET = Properties.APP_SECRET  # consumer secret
OAUTH_TOKEN = Properties.OAUTH_TOKEN
OAUTH_TOKEN_SECRET = Properties.OAUTH_TOKEN_SECRET

SCREEN_NAME = '' # user twitter tag

twitter = Twython(APP_KEY, APP_SECRET,OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
last_id = 10000000000000000000000000000000000000000
init = True
id_data_set = set()

with open(SCREEN_NAME + ".csv", 'w', encoding='utf-8') as mainquery:
    mainquery.write('id,tweet,emoji list\n')
    for i in range(150):
        print('request number: ', i)
        if init:
            tweets = twitter.get_user_timeline(screen_name=SCREEN_NAME,
                                    count=200,  # not guaranteed; maximum number that can be returned
                                    lang='en',
                                    result_type='recent')
        else:
            tweets = twitter.get_user_timeline(screen_name=SCREEN_NAME,
                                    count=200,  # not guaranteed; maximum number that can be returned
                                    lang='en',
                                    max_id=last_id,
                                    result_type='recent')

        for t in tweets:
            id = t['id']
            int_id = int(id)
            if id in id_data_set:
                print('duplicate tweet found', id)
            else:
                if int_id < last_id:
                    last_id = int_id - 1
                text = t['text']
                text = text.replace('\n',' ')
                text = text.replace('\r', ' ')
                # print(text)
                url_pattern = 'https?:\/\/[a-zA-Z0-9@:%._\+~#=\/]*[ ]*'
                url_pattern_re = compile(url_pattern)
                text = sub(url_pattern_re, '', text)
                text = text.replace(',', ' ')
                text = text.strip()
                emoji_list = ''.join(c for c in text if c in emoji.UNICODE_EMOJI)
                text = ''.join(c for c in text if c not in emoji.UNICODE_EMOJI)
                text = text.strip()
                if emoji_list:
                    tweet_data = str(id)+','+str(text) +','+ emoji_list + '\n'
                    mainquery.write(tweet_data)
                    print('text: ', text, 'emoji list: ', emoji_list)
                id_data_set.add(id)
        init = False