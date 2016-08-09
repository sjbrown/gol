
import json
import numpy
from next_iteration import calc_wrap

print('Loading function')

def lambda_handler(event, context):
    print("array is %s", event['array'])
    try:
        result = calc_wrap(event['array']).tolist()
        print("result is %s", result)
        return result
    except Exception as e:
        raise e
