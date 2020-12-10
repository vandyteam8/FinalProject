try:
    import os
    import sys
    import datetime
    import time
    import csv
    import boto3
    import threading
    print("All Modules Loaded ...... ")
except Exception as e:
    print("Error {}".format(e))

class MyDb(object):

    def __init__(self, Table_Name='SimGlucose'):
        self.Table_Name=Table_Name

        self.db = boto3.resource('dynamodb')
        self.table = self.db.Table(Table_Name)

        self.client = boto3.client('dynamodb')

    @property
    def get(self):
        response = self.table.get_item(
            Key={
                'Patient_Id':"1"
            }
        )

        return response

    def put(self, Patient_id='' , Time='', BG='', CGM='', CHO='', insulin='', LBGI='', HBGI='', Risk=''):
        self.table.put_item(
            Item={
                'Patient_id':Patient_id,
                'Time':Time,
                'BG':BG,
                'CGM':CGM,
                'CHO':CHO,
                'insulin':insulin,
                'LBGI':LBGI,
                'HBGI':HBGI,
                'Risk':Risk
            }
        )

    def delete(self, Patient_id='' , Time='', BG='', CGM='', CHO='', insulin='', LBGI='', HBGI='', Risk=''):
        self.table.delete_item(
            Key={
                'Patient_id':Patient_id,
                'Time':Time,
                'BG':BG,
                'CGM':CGM,
                'CHO':CHO,
                'insulin':insulin,
                'LBGI':LBGI,
                'HBGI':HBGI,
                'Risk':Risk
            }
        )

    def describe_table(self):
        response = self.client.describe_table(
            TableName='SimGlucose'
        )
        return response

    @staticmethod
    def sensor_value():

        if time is not None and BG is not None:
            print('time={0:0.1f}*C  BG={1:0.1f}%'.format(temperature, humidity))
        else:
            print('Failed to get reading. Try again!')
        return Time, BG, CGM, CHO, insulin, LBGI, HBGI, Risk


def main():
    global counter

    threading.Timer(interval=1, function=main).start()
    obj = MyDb()
    Temperature , Humidity = obj.sensor_value()
    obj.put(Patient_id=str(counter), Time=str(Time), BG=str(BG))
    counter = counter + 1
    print("Uploaded Sample on Cloud T:{},H{} ".format(Time, BG))


if __name__ == "__main__":
    global counter
    counter = 0
    main()