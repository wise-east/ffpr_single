import mysql.connector
import yaml 
import unittest

class TestMySQLDataBase(unittest.TestCase): 

    def setUp(self): 
        with open('config/config.yaml', 'r') as f: 
            config = yaml.safe_load(f)['mysql']
        self.db = mysql.connector.connect(
            host = config['host'], 
            user = config['user'], 
            passwd = config['passwd'], 
            database = config['database']
        )

    def test_database(self): 
        mycursor = self.db.cursor()
        mycursor.execute("SHOW DATABASES")
        result = [x[0] for x in list(mycursor)]
        self.assertTrue('ffpr' in result) 

if __name__ == "__main__": 
    unittest.main()