#!/usr/bin/env python
# -*- coding:utf-8 -*-

#Module
import twitter
import csv
import sys

#API Input your keys and secrets here
CONSUMER_KEY =  '8PIXARc1gjTkK0epb1BhI46p8'
CONSUMER_SECRET = 'Y4bViqnoMjKMtmVGKJV8HX60HJ3P0ocVWkZN4hEeBdcc4YI72X'
ACCESS_TOKEN = '2563569822-zRH6HkBTm9zOZPhDmXuCzPwStQAs4gmshZsyeNm'
ACCESS_SECRET = 'LZncmwSvIQThxKXFGx0Ry00urgJlLw9b8sTi4Fojm9fLH'

#twitterAPI
api = twitter.Api(consumer_key=CONSUMER_KEY,
                  consumer_secret=CONSUMER_SECRET,
                  access_token_key=ACCESS_TOKEN,
                  access_token_secret=ACCESS_SECRET)

## Key word example (Get Twitter from keyword)
keyword = "World cup"
tweets = api.GetSearch(keyword,count=100)

## Get Twitter from user
user = 34442404 ### Sony account
statuses = api.GetUserTimeline(user)

### Print results
print(tweets[0].text)
print(statuses[1].text)
