from dataclasses import dataclass
from pathlib import Path
import os



@dataclass
class PipelineConfig :
     
    current_directory_Pathlib = Path.cwd() # \end_to_end_ML_projects_\Creditcard_fraud_detection_01
    #base_dir_Pathlib: Path = current_directory_Pathlib / "src" / "components" / "artifacts"

    train_data_path: Path = current_directory_Pathlib / "components" / "artifacts"/ "train.csv"
    train_set_resampled_path: Path = current_directory_Pathlib / "components" / "artifacts"/ "train_resampled.csv"
    test_data_path: Path = current_directory_Pathlib / "components" / "artifacts"/ "test.csv"
    val_data_path: Path = current_directory_Pathlib / "components" / "artifacts"/ "val.csv"
    raw_data_path: Path = current_directory_Pathlib / "components" / "artifacts"/ "data.csv"
    
    preprocessor_path: Path = current_directory_Pathlib  / "components" / "artifacts"/ "preprocessor.pkl"
    trained_model_file_path: Path = current_directory_Pathlib / "components" / "artifacts"/ "model.h5"
       

if __name__=="__main__":
      
      print(PipelineConfig().train_data_path)
     