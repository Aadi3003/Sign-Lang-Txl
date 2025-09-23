WIP
For team member:
After downloading, set up in VSC using the terminal: 

python -m venv venv
.\venv\Scripts\activate         or   .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

Select the proper python interpreter by ctrl+shift+P and select 3.11.7 (venv)
If any other errors pop up, deal with them on the spot - you have all the files, just not the setup.

After making any change, do this on the terminal to upload on GitHub:
(Run the code, so that it saves.)

pip freeze >> requirements.txt
git status
git add .
git ls-files
    if venv appears, remove it by:
    git rm -r --cached venv
    check git ls-files again
git commit -m "<change msg>"
git push origin main
    if error appears, and it tells you to pull:
    git pull origin main --rebase    
    then do the push again

Check if every file on GitHub is matching with your local file.
