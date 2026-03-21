#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
getDiffs.py — Diff File Downloader
Adapted from VUDENC (Wartschinski et al., 2022)
Original: https://github.com/LauraWartschinski/VulnerabilityDetection
Modifications: code modernization (f-strings, error handling)
"""
import requests
import time
import sys
import json
import os

# Get access token
if not os.path.isfile('access'):
    print("Please place a Github access token in this directory.")
    sys.exit()
with open('access', 'r') as accestoken:
    access = accestoken.readline().strip()

# Load mined commits
with open('all_commits.json', 'r') as infile:
    repositories = json.load(infile)

# Load or initialize filter data
datafilter = {}
if os.path.isfile('DataFilter.json'):
    with open('DataFilter.json', 'r') as infile:
        datafilter = json.load(infile)
datafilter.setdefault('showcase', {})
datafilter.setdefault('no-python', {})
datafilter.setdefault('python', {})

print(f"{len(datafilter['showcase'])} repositories are showcases and therefore ignored.")
print(f"{len(datafilter['no-python'])} repositories don't even contain ANY python.")
print(f"{len(datafilter['python'])} might contain python.")

# Load existing data if present
data = {}
if os.path.isfile('PyCommitsWithDiffs.json'):
    with open('PyCommitsWithDiffs.json', 'r') as infile:
        data = json.load(infile)

myheaders = {'Authorization': 'token ' + access}
nopythonlist = {}
total = 0
progress = 0

for repo in repositories:
    progress += 1
    name = repo.split('https://github.com/')[1]

    if name in datafilter['showcase'] or name in datafilter['no-python']:
        print(f"skip: {name}")
        continue

    print(f"\n{repo}     {progress}")

    if repo not in nopythonlist:
        nopythonlist[repo] = {}

    noPythonAtAll = True

    for c in repositories[repo]:
        if repo in data and c in data[repo]:
            continue  # Already processed

        if c in nopythonlist[repo]:
            continue  # Already known non-Python

        target = repo + '/commit/' + c + '.diff'
        time.sleep(0.2)  # Respect rate limits

        try:
            response = requests.get(target, headers=myheaders)
            diffcontent = response.content.decode('utf-8', errors='ignore')
        except Exception as e:
            print("An exception occurred. Skipping.")
            continue

        if ".py" in diffcontent:
            noPythonAtAll = False
            if repo not in data:
                data[repo] = {}
            data[repo][c] = repositories[repo][c]
            data[repo][c]["diff"] = diffcontent
            total += 1
        else:
            nopythonlist[repo][c] = {}

    if noPythonAtAll:
        datafilter['no-python'][name] = {}
    else:
        datafilter['python'][name] = {}

    # Save progress regularly
    if progress % 100 == 0:
        print("Saving progress...")
        with open('DataFilter.json', 'w') as outfile:
            json.dump(datafilter, outfile)
        with open('PyCommitsWithDiffs.json', 'w') as outfile:
            json.dump(data, outfile)

print(f"{total} commits modifying Python were found.")

# Final save
with open('DataFilter.json', 'w') as outfile:
    json.dump(datafilter, outfile)
with open('PyCommitsWithDiffs.json', 'w') as outfile:
    json.dump(data, outfile)
