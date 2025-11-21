import logging

from service_functions import send_telegramm_message

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

send_telegramm_message('Greetings, Master!')

