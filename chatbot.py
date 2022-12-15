from joblib import load
import numpy as np
import random
import time

model = load("model.joblib")
clf = model["clf"]
replies = model["tags_and_replies"]
vectorizer = model["vectorizer"]

tag_to_idx = {}
for i, tag in enumerate(clf.classes_):
    tag_to_idx[tag] = i

print("Hi, Welcome to chatbot!")
print("When you're bored type exit.")
print("Let's start!\n")

while True:
    print("you: ", end="")
    question = input()
    vector = vectorizer.transform([question])
    tag = clf.predict(vector)[0]
    if clf.predict_proba(vector)[0, tag_to_idx[tag]] > 0.5:
        if tag == "exit":
            print("bot: Ok, exiting..")
            break
        responses = replies[tag]
        idx = random.randint(0, len(responses)-1)
        print("bot:", responses[idx])
    else:
        print("bot: I don't understand. Could you please rephrase it?")
