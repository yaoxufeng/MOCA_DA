import logging

def setup_global_logger(name, log_dir):
    logging.basicConfig(level=logging.INFO,
                        filename='{}/{}.log'.format(log_dir, 'run'),
                        filemode='a')
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s', datefmt='%m-%d %H:%M')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger
