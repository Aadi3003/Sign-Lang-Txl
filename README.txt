After downloading, set up in VSC using the terminal: 

python -m venv venv
.\venv\Scripts\activate 
pip install -r requirements.txt

After making any change, do this on the terminal to upload on GitHub:

Run the code, so that it saves.
pip freeze >> requirements.txt

git status
git add .
git ls-files
    if venv appears, remove it
    git rm -r --cached venv
    check git ls-files again
git commit -m "<change msg>"
git push origin main
    if error appears, and it tells you to pull:
    git pull origin main --rebase    
    then do the push again

Check if every file on GitHub is matching with your local file.
