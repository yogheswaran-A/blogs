# First time
## Steps
### 1. Folder structure: 
1.                    blogs(outside)
                         -blogs(inside)
                                - _build
                                - data
                                - other files
                        - .github
                        - readme.md
                        - requirements.txt
                        - others

### 2. build your note book, cd into blogs(outside): jupyter-book build blogs/ , remember if you are re-runing this command make sure to delete the _build folder
### 3. create a repo called blogs online, copy all the files inside blogs(outside) in local into github repo blogs, which I will call as blogs(outside)
### 4. cd into blogs(outside) folder
### 5. git add ./*
### 6. git commit -m "adding my first book!"
### 7. git push
### 8. pip install ghp-import
### 9. cd into  blogs(inside) folder, run cmd:  ```ghp-import -n -p -f _build/html``` , It will show merge and pull. No need to do anything.
### 10. go to settings, Pages.
1) Under Source, select deploy from brach. 
2) Under branch select gh-pages, select /root and save.
### 11. Under action see if it is deployed.

# Second time
## Steps
### 1. cd into blogs(outside) folder
### 2. delete the _build folder inside the blogs(inside)
### 2. from the blogs(outside) build your note book: jupyter-book build blogs/
### 3. cd into  blogs(outside)
### 3. git add ./*
### 4. git commit -m "adding my first book!"
### 5. git push
### 6. cd into  blogs(inside) folder, run cmd: ghp-import -n -p -f _build/html
### 7. Under action see if it is deployed.

# If you face an problem check the build error. I once changed the config file etc to modify the github actions flow using cockied cutter. Instead of this, follow the instructions in jupyter book and download the sample notebook and modify it accordingly.