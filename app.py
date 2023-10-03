from flask import Flask, render_template, request
import pandas as pd
import sys  # Ensure this is imported
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.exception import CustomException
from src.utils import *
from pathlib import Path  # Import Path from pathlib
from src.logger import logging


class FilePathConfig:
    BASE_DIR: Path = Path(__file__).parent
    VAL_CSV_PATH: Path = BASE_DIR / "src" / "components" / "artifacts" / "val.csv"

class FlaskApp:

    def __init__(self):
        self.file_path_config = FilePathConfig()

    def initiate_flask_app(self):

        app = Flask(__name__)

        @app.route('/', methods=['GET', 'POST'])
        def index():
            prediction = None
            sample = None

            if request.method == 'POST':
                try:
                    data_dict = request.form.to_dict()
                    data_df = pd.DataFrame([data_dict])
                    pipeline = PredictPipeline()
                    predictions = pipeline.predict(data_df)
                    prediction = "Fraud" if predictions[0] == 1 else "Not Fraud"
                except Exception as e:
                    raise CustomException(e, sys)

            sample_data = pd.read_csv(str(self.file_path_config.VAL_CSV_PATH)).iloc[0] 
            logging.info((str(self.file_path_config.VAL_CSV_PATH)).iloc[0])
            sample = sample_data.to_dict()

            return render_template('index.html', prediction=prediction, sample=sample)
        
        @app.route('/Predictdata')
        
        
        

        if __name__ == '__main__':
            app.run(debug=True)

# Example of how to run the app:
if __name__ == '__main__':
    app_instance = FlaskApp()
    app_instance.initiate_flask_app()
