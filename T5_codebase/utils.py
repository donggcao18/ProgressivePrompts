import os
import datetime
import logging

def set_logger(log_dir='logs'):
    # Reset root logger 
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    os.makedirs(log_dir, exist_ok=True)
    vietnam_tz = datetime.timezone(datetime.timedelta(hours=7))  # UTC+7 for Vietnam
    log_file = os.path.join(log_dir, f"datetime_{datetime.datetime.now(vietnam_tz).strftime('%Y-%m-%d_%H-%M-%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )