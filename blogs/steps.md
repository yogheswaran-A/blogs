# First time
## Folder structure: 
###                    blogs(outside)
###                         -blogs(inside)
###                                - _build
###                                - data
###                                - other files
###                        - .github
###                        - readme.md
###                        - requirements.txt
###                        - others

## build your note book, cd into blogs(outside): jupyter-book build mybookname/
## create a repo called blogs online, copy all the files inside blogs(outside) in local into github repo blogs, which I will call as blogs(outside)
## cd into blogs(outside) folder
## git add ./*
## git commit -m "adding my first book!"
## git push
## pip install ghp-import
## run cmd:  ghp-import -n -p -f _build/html , It will show merge and pull. No need to do anything.
## go to settings, Pages. Under Source,1) select deploy from brach. 2) Under branch select gh-pages, select /root and save.
## Under action see if it is deployed.

# Second time
##  build your note book(outside): jupyter-book build mybookname/
## cd into blogs(outside) folder
## git add ./*
## git commit -m "adding my first book!"
## git push