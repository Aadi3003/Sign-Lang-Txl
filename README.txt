After downloading, set up in VSC using the terminal: 

python -m venv venv
.\venv\Scripts\activate 
pip install -r requirements.txt

After making any change, do this on the terminal to upload on GitHub:

git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YourUsername/your-repo.git
git push -u origin main
