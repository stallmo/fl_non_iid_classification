import os

# determine current directory
this_dir = os.path.abspath(os.path.dirname(__file__))
PATH_TO_DATA_DIR = os.path.join(this_dir, '../../../../data')
PATH_TO_DATA_CENTRALIZED_DIR = os.path.join(PATH_TO_DATA_DIR, 'centralized')
PATH_TO_DATA_DISTRIBUTED_DIR = os.path.join(PATH_TO_DATA_DIR, 'distributed')
PT_DATALOADER_NUM_WORKERS = 2