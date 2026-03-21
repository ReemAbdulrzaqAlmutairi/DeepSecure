#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
getData.py — Source Code Extractor
Adapted from VUDENC (Wartschinski et al., 2022)
Original: https://github.com/LauraWartschinski/VulnerabilityDetection
Modifications: mode and keywords updated for new vulnerability types
"""
import myutils
import time
import sys
import json
import subprocess
from datetime import datetime
import requests 
import pickle
from pydriller import RepositoryMining

def getChanges(rest):
    ### extracts the changes from the pure .diff file
    ### start by parsing the header of the diff

    changes = []
    while "diff --git" in rest:
        filename = ""

        start = rest.find("diff --git") + 1
        secondpart = rest.find("index") + 1

        # get the title line which contains the file name
        titleline = rest[start:secondpart]

        if not ".py" in rest[start:secondpart]:
            # No python file changed in this part of the commit
            rest = rest[secondpart+1:]
            continue

        # calculate end of this diff block
        end_rel = rest[start:].find("diff --git")
        if end_rel != -1:
            end = start + end_rel
            filechange = rest[start:end]
            rest = rest[end:]
        else:
            filechange = rest[start:]
            rest = ""

        filechangerest = filechange

        # split hunks by "@@"
        hunks = filechangerest.split("@@")[1:]
        for h in hunks:
            hunk = h.split("@@", 1)[0].strip()

            # skip header lines
            if ("class" in hunk or "def" in hunk) and "\n" in hunk:
                hunk = hunk[hunk.find("\n"):].strip()

            if len(hunk) > 0:
                changes.append([titleline, hunk])

    return changes

def getFilename(titleline):
    #extracts the file name from the title line of a diff file
    s = titleline.find(" a/")+2
    e = titleline.find(" b/")
    name = titleline[s:e]

    if titleline.count(name) == 2:
        return name
    elif ".py" in name and (" a"+name+" " in titleline):
        return name
    else:
        print("couldn't find name")
        print(titleline)
        print(name)

def makechangeobj(changething):
    #for a single change, consisting of titleline and raw code, create a usable object by extracting all relevant info

    change = changething[1]
    titleline = changething[0]

    if "<html" in change:
        return None

    if "sage:" in change or "sage :" in change:
        return None

    thischange = {}

    if myutils.getBadpart(change) is not None:      
        badparts = myutils.getBadpart(change)[0]
        goodparts = myutils.getBadpart(change)[1]
        linesadded = change.count("\n+")
        linesremoved = change.count("\n-")
        thischange["diff"] = change
        thischange["add"] = linesadded
        thischange["remove"] = linesremoved
        thischange["filename"] = getFilename(titleline)
        thischange["badparts"] = badparts
        thischange["goodparts"] = []
        if goodparts is not None:
            thischange["goodparts"] = goodparts
        if thischange["filename"] is not None:
            return thischange

    return None

#===========================================================================

#load list of all repositories and commits
with open('PyCommitsWithDiffs.json', 'r') as infile:
    data = json.load(infile)

now = datetime.now() # current date and time
nowformat = now.strftime("%H:%M")
print("finished loading ", nowformat)

progress = 0
changedict = {}

for mode in ["broken_authentication"]:
    if mode == "broken_authentication":
        allowedKeywords = [            
            "broken authentication",
            "authentication bypass",
            "auth bypass",
            "login vulnerability",
            "password flaw",
            "authentication fix",
            "session hijacking",
            "weak password",
            "unauthorized login",
            "fix auth",
            "credential exposure"
        ]

    suspiciouswords = ["injection", "vulnerability", "exploit", " ctf","capture the flag","ctf","burp","capture","flag","attack","hack"]

    badwords = ["sqlmap", "sql-map", "sql_map","ctf "," ctf"]

    progress = 0
    datanew = {}

    for r in data:
        print(f"  → Processing [{progress+1}/{len(data)}]: {r}")
        progress += 1

        suspicious = False
        for b in badwords:
            if b.lower() in r.lower():
                suspicious = True
        if suspicious:
            continue

        if("anhday22" in r or "Chaser-wind" in r or "/masamitsu-murase" in r or "joshc/young-goons" in r or "notakang" in r or "sudheer628" in r or "mihaildragos" in r or "aselimov" in r or "tamhidat-api" in r or "aiden-law" in r or "sreeragvv" in r or "LaurenH1090" in r or "/matthewdenaburg1" in r or "haymanjoyce" in r or "/bloctavius" in r or "jordanott/No-Weight-Sharing" in r or "bvanseg" in r or "sudoku-solver" in r or "tgbot" in r or "lluviaBOT" in r or "jumatberkah" in r or "luisebg" in r or "emredir" in r or "anhday22" in r or "faprioryan" in r or "pablogsal" in r or "zhuyunfeng111" in r or "bikegeek/METplus" in r or "chasinglogic" in r or "Sudhir0547" in r or "fyp_bot" in r):
            continue

        changesfromdiff = False
        changeCommits = []
        for c in data[r]:
            irrelevant = True
            for k in allowedKeywords:
                if k.lower() in data[r][c]["keyword"].lower():
                    irrelevant = False

            if irrelevant:
                continue

            if not (".py" in data[r][c]["diff"]):
                continue

            if not "message" in data[r][c]:
                data[r][c]["message"] = ""

            if not c in changedict:
                changedict[c] = 0
            changedict[c] += 1
            if changedict[c] > 5:
                continue

            changes = getChanges(data[r][c]["diff"])

            for change in changes:
                thischange = makechangeobj(change)

                if thischange is not None:
                    if not "files" in data[r][c]:
                        data[r][c]["files"] = {}
                    f = thischange["filename"]

                    if f is not None:
                        suspicious = False
                        for s in suspiciouswords:
                            if s.lower() in f.lower():
                                suspicious = True

                        if not suspicious:   
                            if not f in data[r][c]["files"]:
                                data[r][c]["files"][f] = {}
                            if not "changes" in data[r][c]["files"][f]:
                                data[r][c]["files"][f]["changes"] = []
                            data[r][c]["files"][f]["changes"].append(thischange)
                            changesfromdiff = True
                            changeCommits.append(c)

        if changesfromdiff:
            print("\n\n" + mode + "    mining "  + r + " " + str(progress) + "/" + str(len(data)))

            commitlist = []
            try:
                for commit in RepositoryMining(path_to_repo=r).traverse_commits():
                    commitlist.append(commit.hash)

                    if not commit.hash in changeCommits:
                        continue

                    for m in commit.modifications:
                        if m.old_path != None and m.source_code_before != None:
                            if not ".py" in m.old_path:
                                continue

                            if len(m.source_code_before) > 30000:
                                continue

                            for c in data[r]: 
                                if c == commit.hash:
                                    print("  found commit " + c)
                                    if not "files" in data[r][c]:
                                        print("  no files :(")

                                    data[r][c]["msg"] = commit.msg
                                    for badword in badwords:
                                        if badword.lower() in commit.msg.lower():
                                            suspicious = True
                                    if suspicious:
                                        print("  suspicious commit msg: \"" + commit.msg.replace("\n"," ")[:300] + "...\"")
                                        continue

                                    for f in data[r][c]["files"]:
                                        if m.old_path in f:
                                            if not ("source" in data[r][c]["files"][f] and len(data[r][c]["files"][f]["source"]) > 0):
                                                sourcecode = "\n" + myutils.removeDoubleSeperatorsString(myutils.stripComments(m.source_code_before))
                                                data[r][c]["files"][f]["source"] = sourcecode

                                            if not ("sourceWithComments" in data[r][c]["files"][f] and len(data[r][c]["files"][f]["sourceWithComments"]) > 0):
                                                data[r][c]["files"][f]["sourceWithComments"] = m.source_code_before

                                            if not ("sourceWithComments" in data[r][c]["files"][f] and len(data[r][c]["files"][f]["sourceWithComments"]) > 0):
                                                data[r][c]["files"][f]["sourcecodeafter"] = ""
                                                if m.source_code is not None:
                                                    data[r][c]["files"][f]["sourcecodeafter"] = m.source_code

                                            if not r in datanew:
                                                datanew[r] = {}
                                            if not c in datanew[r]:
                                                datanew[r][c] = {}

                                            datanew[r][c] = data[r][c]
                                            print("     ->> added to the dataset.")

            except Exception as e:
                print("Exception occured.")
                print(e)
                time.sleep(2)
                continue

    print("done.")
    print(len(data))

    
