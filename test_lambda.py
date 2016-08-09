#! /usr/bin/env python

import json
import requests

url = 'https://ib4dch6ahk.execute-api.us-east-1.amazonaws.com/prod/calc_wrap'

test_matrix = [
    [0,0,0],
    [0,1,0],
    [0,0,0]
]

resp = requests.put(url, data=json.dumps({'array': test_matrix}))

print resp.text
