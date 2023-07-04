from src.models import run_models
from src.run_live import run_live

if __name__ == "__main__":
    
    filename = 'resources/dataset_seconds.csv'
    #filename = 'resources/dataset_repeated.csv'
    
    run_models(filename)
    #run_live()