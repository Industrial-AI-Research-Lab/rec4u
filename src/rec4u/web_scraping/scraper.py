import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from typing import List
import yaml
from yaml import Loader
import warnings

options = Options()
options.headless = True
options.add_argument("--window-size=1920,1200")

if 'PATH_TO_CRAWLER_CONFIG' in os.environ:
    filepath = os.environ['PATH_TO_CRAWLER_CONFIG']
else:
    raise FileNotFoundError

with open(filepath, 'r') as file:
    config = yaml.load(file)

class Scraper:
    def __init__(self):
        driver = webdriver.Chrome(options=options, executable_path=config)

        # google, yandex sites
        driver.get("http://www.google.com")
        raise NotImplementedError

    def parse_config(self):
        pass

    def keywords_searcher(self, keywords: List):
        raise NotImplementedError
