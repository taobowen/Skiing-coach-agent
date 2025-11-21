# agent_lambda.py
from mangum import Mangum
from agent_server import app

handler = Mangum(app)
