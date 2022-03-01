from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from typing import List

class Scraper:
    def __init__(self):
        raise NotImplementedError

    def keywords_searcher(self, keywords: List):

        raise NotImplementedError