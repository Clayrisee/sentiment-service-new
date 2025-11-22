# Implement dynamo_db -> pertemuan 5
import boto3
import uuid
import pandas as pd
from .utils import convert_floats_to_decimals
from loguru import logger

class UncertaintyDynamoDB:
    def __init__(self, table_name: str) -> None:
        logger.info(f"Setting UncertaintyDynamoDB with name: {table_name}")
        self.dynamo_db = boto3.resource('dynamodb')
        self.uncertainty_table = self.dynamo_db.Table(table_name)
        self.table_name = table_name
    
    def write_event(self, data: dict):
        try:
            logger.info(f"Write event: {data} to Table: {self.table_name}")
            data = convert_floats_to_decimals(data)
            response = self.uncertainty_table.put_item(
                Item={
                    'id': str(uuid.uuid4()),
                    'timestamp': data['timestamp'], 
                    'text': data['text'], 
                    'prediction': data['prediction'], 
                    'confidence': str(data['confidence']),
                    'uncertainty_score': data['uncertainty_score'],
                    'is_uncertain': data['is_uncertain'] 
                }
            )
            logger.info(f"Write Event Success. Response:{response}")
        except Exception as e:
            logger.error(f"Error write event. Error: {e}")

    def fetch_data(self) -> pd.DataFrame:
        try:
            logger.info(f"Fetching data from Table: {self.table_name}")
            items = []
            response = self.uncertainty_table.scan()
            items.extend(response.get('Items', []))

            # Handle pagination
            while 'LastEvaluatedKey' in response:
                response = self.uncertainty_table.scan(
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                items.extend(response.get('Items', []))

            logger.info(f"Fetched {len(items)} items from Table: {self.table_name}")
            return pd.DataFrame(items)
        except Exception as e:
            logger.error(f"Error fetching data. Error: {e}")
            return pd.DataFrame()