import sys

import logging
logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", "%H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)

debug_exceptions = False
