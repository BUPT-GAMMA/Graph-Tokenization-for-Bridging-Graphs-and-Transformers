import optuna
from optuna_dashboard import run_server
from optuna.storages import JournalStorage, JournalFileStorage




if __name__ == "__main__":
    storage = JournalStorage(JournalFileStorage('hyperopt/journal/finetune_with_options.db'))


    # Start the Optuna Dashboard server on localhost:8080
    run_server(storage, host="localhost", port=18080)