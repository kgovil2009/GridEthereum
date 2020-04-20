import psycopg2 as p
import psycopg2.extras as pe
from configuration import cfg as c

"""This file takes the credentials from configuration.py and connects with Amazon RDS instance. The postgres db 
instance has the data for ponzi and non ponzi schemes. """

connection = p.connect(dbname=c.dbname, user=c.user, host=c.host, port=c.port, password=c.password)

class GetData:
    def getPonziData(self):
        cursor_ponzi = connection.cursor()
        cursor_ponzi.execute("select Address from ethereum.ponzi_anomaly where ponzi_anomaly.label =  '{Ponzi}'")
        rows_ponzi = cursor_ponzi.fetchall()
        return rows_ponzi

    def getNonPonziData(self):
        cursor_nonponzi = connection.cursor()
        cursor_nonponzi.execute("select Address from ethereum.ponzi_anomaly where ponzi_anomaly.label =  '{Nonponzi}'")
        rows_nonponzi = cursor_nonponzi.fetchall()
        return rows_nonponzi





