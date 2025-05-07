import json
import os

class Best_params_handler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.params = self.load_params()

    def load_params(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                return json.load(f)
        else:
            return {}

    def save_params(self):
        with open(self.file_path, "w") as f:
            json.dump(self.params, f, indent=4)

    def get_params(self, months, currency):
        month_str = str(months)
        return self.params.get(month_str, {}).get(currency, None)

    def set_params(self, months, currency, params):
        month_str = str(months)
        if month_str not in self.params:
            self.params[month_str] = {}
        self.params[month_str][currency] = params
        self.save_params()