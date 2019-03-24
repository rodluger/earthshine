import subprocess
import os
hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8")[:-1]
slug = os.getenv("TRAVIS_REPO_SLUG", "user/repo")
with open("gitlinks.tex", "w") as f:
    print(r"\newcommand{\codelink}[1]{\href{https://github.com/%s/blob/%s/notebooks/#1.ipynb}{\codeicon}\,\,}" % (slug, hash), file=f)
