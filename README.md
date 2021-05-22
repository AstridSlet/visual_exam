# Project portfolio: Visual analytics

## Repo structure

This repository includes four projects including three class assignments developed during the course and one final project. The four four projects largely has the following folder-structure:

| Folder | Description|
|--------|:-----------|
```project_folder``` | project folder.
```/src``` | folder holding main scripts.
```/data```| data folder holding sample datasets used for the project.
```/utils```| folder holding utility functions used in main scripts.
```/output``` | output folder holding an example output, produced when running the script with default settings.


## Technicalities

To run the scripts for the projects of this repository I recommend cloning this repository and installing required dependencies in a virtual environment:

```
$ git clone https://github.com/AstridSlet/visual_exam.git
$ cd visual_exam
$ bash create_venv.sh
$ source visual_venv/bin/activate
```

If you run into issues with libraries/dependencies when running any of the scripts, the necessary packages can be installed manually using the following command line code:

```
$ cd visual_exam
$ source visual_venv/bin/activate
$ pip install {library_name}
$ deactivate
```

When you are done using the virtual environment, you can remove it with: 

```
$ rm -rf visual_venv
```

## Contact details
If you experience any issues with downloading the packages and installing dependencies feel free to contact me by e-mail: astrid.rybner@hotmail.com


## Acknowledgements
Credits to [Ross](https://pure.au.dk/portal/en/persons/ross-deans-kristensenmclachlan(29ad140e-0785-4e07-bdc1-8af12f15856c).html) for repo structure inspiration and the utility functions developed in class used in original/adapted versions for these projects.  


