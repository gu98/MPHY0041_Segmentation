
import os
from tensorflow.keras.utils import get_file
import zipfile


'''
# download data
os.system('wget https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/-/archive/promise12/datasets-promise12.zip')
os.system('mkdir data')
os.system('unzip datasets-promise12.zip -d ./data')
os.system('rm datasets-promise12.zip')
os.system('mkdir result')
'''

DATA_PATH = './data'
RESULT_PATH = './result'

os.makedirs(DATA_PATH)
temp_file = get_file(fname='datasets-promise12.zip',
                     origin='https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/-/archive/promise12/datasets-promise12.zip')
with zipfile.ZipFile(temp_file,'r') as zip_obj:
    zip_obj.extractall(DATA_PATH)
os.remove(temp_file)
os.makedirs(RESULT_PATH)

print('Promise12 data downloaded: <%s>.' % os.path.abspath(os.path.join(DATA_PATH,'datasets-promise12')))
print('Result directory created: <%s>.' % os.path.abspath(RESULT_PATH))

