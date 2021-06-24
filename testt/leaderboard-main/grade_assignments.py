from sklearn.metrics import mean_absolute_error, accuracy_score
import sklearn, json, sys, os
import pandas as pd


import signal
from contextlib import contextmanager


@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)

    try:
        yield
    except TimeoutError:
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def raise_timeout(signum, frame):
    raise TimeoutError


# WORK_DIR = '/app'

WORK_DIR = '/home/zeyuy/auto_grading_nlp'
# WORK_DIR = '/home/zeyuy/grading_file'

challenge_folder = WORK_DIR + "/challenges"
# challenge_folder = "challenges/"
df = pd.read_csv(WORK_DIR + '/scores.csv', header=[0,1], index_col=0)
team_name = sys.argv[1]
# df.loc[team_name]=0

if team_name in df.index:
    i=2
    while team_name+'_'+str(i) in df.index:
        i+=1
    team_name = team_name+'_'+str(i)

df.loc[team_name]=0
idx=pd.IndexSlice
df.loc[team_name, idx[:,'MAE']] = float('nan')

results = []

for i in range(1, 11):
    with timeout(90):
        challenge = os.path.join(challenge_folder, "yelp_challenge_"+str(i)+".jsonl")
        os.system("CUDA_VISIBLE_DEVICES=2 python test_submission.py "+challenge)

        answers_file = os.path.join(challenge_folder, "yelp_challenge_"+str(i)+"_with_answers.jsonl")
        answers = []
        with open(answers_file, "r") as f:
            for line in f:
                answers.append(json.loads(line))

        predictions = {}
        with open("./output.jsonl", "r") as f:
            for line in f:
                pred = json.loads(line)
                predictions[pred['review_id']] = pred['predicted_stars']


        answer_list, pred_list = [], []
        for answer in answers:
            if answer['review_id'] not in predictions:
                print("MISSING REVIEW ID: "+answer['review_id'])
                continue
            answer_list.append(answer['stars'])
            pred_list.append(predictions[answer['review_id']])

        MAE = mean_absolute_error(answer_list, pred_list)
        ACC = accuracy_score(answer_list, pred_list)
        results += [str(MAE), str(ACC)]
        df.loc[team_name,("challange"+str(i), "MAE")]=MAE
        df.loc[team_name,("challange"+str(i), "Accuracy")]=ACC
        print("Challenge %d. Size: %d; MAE: %.4f, ACC: %.4f" % (i, len(answer_list), MAE, ACC))
        print("---")
df.to_csv(WORK_DIR +'/scores.csv')
print("FINAL")
print(",".join(results))