from pydantic import BaseModel
from pymongo import MongoClient
from secrets import MONGO_URI
from abc import ABC, abstractmethod

class Database(ABC):
    """
    Abstract Base Class
    """
    def __init__(self, type : str) -> None:
        self.type = type

    @abstractmethod
    def db_insert_query(self, query_text: str, domain: str, intent_detected: str):
        pass


class MongoDatabase(Database):
    def __init__(self) -> None:
        super().__init__("MongoDB")
        self.client = MongoClient(MONGO_URI)
        self.db = self.client['BTP_Project']
        self.collection = self.db['queries']
    
    def db_insert_query(self, query_text: str, domain: str, intent_detected: str):
        return self.collection.insert_one({
            "query_text": query_text,
            "domain": domain,
            "intent_detected": intent_detected
        })


class TextDatabase(Database):
    def __init__(self, dir : str) -> None:
        super().__init__("Text")
        
        if dir[-1] != "/":
            dir += "/"
        self.dir = dir
        self.file_num = 0
        self.line_count = 0
        self.file_line_limit = 1000

    def db_insert_query(self, query_text: str, domain: str, intent_detected: str):
        if self.line_count > self.file_line_limit:
            self.file_num += 1
            self.line_count = 0

        file = open(self.dir + f"logs{self.file_num}.tsv", "a")
        if self.line_count == 0:
            file.write(f"query_text\tdomain\tintent_detected\n")

        file.write(f"{query_text}\t{domain}\t{intent_detected}\n")
        file.close()
        self.line_count += 1
        


class TextQuery(BaseModel):
    query_text: str
