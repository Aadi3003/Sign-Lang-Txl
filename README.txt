After downloading, set up in VSC using the terminal: 

python -m venv venv
.\venv\Scripts\activate 
pip install -r requirements.txt

After making any change, do this on the terminal to upload on GitHub:

git status
git add .
git ls-files
    if venv appears, remove it
    git rm -r --cached venv
    check git ls-files again
git commit -m "<change msg>"
git push
