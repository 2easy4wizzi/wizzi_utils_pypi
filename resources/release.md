# Python packages
[based on this manual](https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56)

##  update a version: 
* select new_v > old_v<br />
e.g. old_v = 2.1.3, new_v = 2.1.4 or 2.2.0 or 3.0.0

## setup.py:
* update fields version and download_url with new_v  
* change meta-data if needed (install_requires, keywords, md...)

## version() in misc_tools.py
* update to new_v

## github
1. releases
2. draft new release
3. tag new_v 
4. publish release and copy release url of tar.gz

## creating new dist
```bash
# in root dir (e.g. (wu)2021wizzi_utils):
python setup.py sdist
twine upload dist/*
username = 2easy4wizzi
password = XXXX
pip install wizzi_utils --upgrade
```