"""This is the main package for my custom package."""
import sys
sys.path.append(r'C:\DataScience\Visual studio\end_to_end_ML_projects_\Creditcard_fraud_detection_01')

# Import key modules and classes
from src.exception import *
from .logger import logging

from .utils import df_balancer, save_object
