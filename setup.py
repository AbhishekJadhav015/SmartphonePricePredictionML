from typing import List
from setuptools import find_packages , setup

#This function will return list of requirements
def get_requirements(file_path:str)-> List[str]: 
    '''
    This function will return list of requirements
    '''
    HYPEN_E_DOT = "-e ."
    requiremnets =[]
    with open(file_path) as file_obj :
        requiremnets = file_obj.readlines()
        requiremnets= [req.replace("\n","").strip() for req in requiremnets]
        
        if HYPEN_E_DOT in requiremnets:
            requiremnets.remove(HYPEN_E_DOT)
        
    return requiremnets
    
    
#Basic Setup    
setup(
    name="ML_PROJECT",
    version ="0.0.1" ,
    author ="Abhishek" ,
    author_email = "abhishekjadhav0015@gmail.com" ,
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt") ,
    
)