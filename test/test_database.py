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

    def test_database_exists(self): 
        mycursor = self.db.cursor()
        mycursor.execute("SHOW DATABASES")
        result = [x[0] for x in list(mycursor)]

        # check if the correct database schema exists
        self.assertTrue('ffpr' in result) 
        mycursor.execute("SELECT * FROM main")

        # check if the right column names exist 
        self.assertTrue('title' in mycursor.column_names and 'article' in mycursor.column_names)

    def test_insert_to_database(self): 
        mycursor = self.db.cursor() 
        mycursor.execute('SELECT * FROM main')
        results = mycursor.fetchall() 

        title = "sample title"
        article = "sample article"
        sql = "INSERT INTO main (title, article) VALUES (%s, %s)"
        val = (title, article)
        mycursor.execute(sql, val)
        mycursor.execute('SELECT * FROM main')
        new_results = mycursor.fetchall() 
        
        # check that new results have one more row than previous result
        self.assertTrue(len(results) + 1 == len(new_results))
        
        # check that the title and article is in the newly added row
        added_row = new_results[-1]
        self.assertTrue(title in added_row and article in added_row)


if __name__ == "__main__": 
    unittest.main()