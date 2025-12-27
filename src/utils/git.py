import subprocess

import git


def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"


def commit(content):
    repo = git.Repo(search_parent_directories=True)
    try:
        g = repo.git
        g.add("--all")
        res = g.commit("-m " + content)
        print(res)
    except Exception as e:
        print("no need to commit")


import datetime

import git_util

if __name__ == "__main__":
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git_util.commit("RUN_" + date_str)
    ...
