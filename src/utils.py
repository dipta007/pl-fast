import git


def get_git_commit_hash():
    repo = git.Repo(search_parent_directories=True)
    commit_hash = repo.head.object.hexsha
    return commit_hash