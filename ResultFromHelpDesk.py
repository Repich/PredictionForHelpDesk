# -*- coding: utf-8 -*-
import pyodbc
import datetime
import os


def get_single_incident(number):

    results = {'result':'False'}

    cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=MyServer;DATABASE=MyDataBase;UID=MyLogin;PWD=MyPassword')
    cursor = cnxn.cursor()
    sql = '''
    select Number, ShortDescription, Description, Priority, Status_New, Status_Service
    FROM [MyDataBase].[dbo].[OP_Request] (nolock) where Number = '''+str(number)

    try:
        cursor.execute(sql)
        columns = [column[0] for column in cursor.description]
        for row in cursor.fetchall():
            results['result'] = 'True'
            results.update(dict(zip(columns, row)))
    except Exception as e:
        print('Exception')
    return (results)